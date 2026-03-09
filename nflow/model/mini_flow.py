import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class Bijection(nn.Module):
    def __init__(self):
        super(Bijection, self).__init__()

    def forward(self, x):
        pass

    def inverse(self, z):
        pass

class ActNorm(Bijection):
    '''
    Base class for activation normalization [1].
    References:
        [1] Glow: Generative Flow with Invertible 1×1 Convolutions,
            Kingma & Dhariwal, 2018, https://arxiv.org/abs/1807.03039
    '''

    def __init__(self, num_features, data_dep_init=True, eps=1e-6):
        super(Bijection, self).__init__()
        self.num_features = num_features
        self.data_dep_init = data_dep_init
        self.eps = eps

        self.register_buffer('initialized', torch.zeros(1) if data_dep_init else torch.ones(1))
        self.register_parameter('shift', nn.Parameter(torch.zeros(1, self.num_features)))
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, self.num_features)))

    def data_init(self, x):
        self.initialized += 1.
        with torch.no_grad():
            x_mean, x_std = self.compute_stats(x)
            self.shift.data = x_mean
            self.log_scale.data = torch.log(x_std + self.eps)

    def forward(self, x):
        if self.training and not self.initialized: self.data_init(x)
        z = (x - self.shift) * torch.exp(-self.log_scale)

        ldj = torch.sum(-self.log_scale).expand([x.shape[0]])
        return z, ldj

    def inverse(self, z):
        return self.shift + z * torch.exp(self.log_scale)


    def compute_stats(self, x):
        '''Compute x_mean and x_std'''
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True)
        return x_mean, x_std

    def ldj_multiplier(self, x):
        '''Multiplier for ldj'''
        return x.shape[2]


class Conv1x1(Bijection):

    def __init__(self, input_dim):
        super(Conv1x1, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.orthogonal_(self.weight)
        # self._inverse = torch.inverse(self.weight)

    def forward(self, x):
        z = F.conv1d(x.unsqueeze(-1), self.weight.unsqueeze(-1)).squeeze(-1)
        ldj = (torch.slogdet(self.weight)[1]).expand([x.shape[0]])
        # self._inverse = torch.inverse(self.weight)
        return z, ldj

    def inverse(self, z):
        return F.conv1d(z.unsqueeze(-1), self._inverse.unsqueeze(-1)).squeeze(-1)


# class Conv1x1(Bijection):
#     def __init__(self, input_dim):
#         super(Conv1x1, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
#         nn.init.orthogonal_(self.weight)
#
#     def forward(self, x, cond=None):
#         z = F.conv1d(x.unsqueeze(-1), self.weight.unsqueeze(-1)).squeeze(-1)
#         ldj = (torch.slogdet(self.weight)[1]).expand([x.shape[0]])
#         return z, ldj
#
#     def inverse(self, z):
#         _inverse = torch.inverse(self.weight)
#         return F.conv1d(z.unsqueeze(-1), _inverse.unsqueeze(-1)).squeeze(-1)


class SwitchSides(Bijection):

    def forward(self, x):
        x1, x2 = torch.chunk(x, dim=1, chunks=2)
        y = torch.cat((x2, x1), 1)
        return y, 0

    def inverse(self, z):
        x1, x2 = torch.chunk(z, dim=1, chunks=2)
        x = torch.cat((x2, x1), 1)
        return x

class AffineCouplingLayer(Bijection):
    def __init__(self, net):
        super(AffineCouplingLayer, self).__init__()
        self.net = net

    def forward(self, x):
        x1, x2 = torch.chunk(x, dim=1, chunks=2)
        log_s, t = self.net(x1)
        log_s = log_s.tanh()
        y1 = x1
        y2 = torch.exp(log_s) * x2 + t
        y = torch.cat((y1, y2), 1)
        log_det = log_s.sum(1)
        return y, log_det

    def inverse(self, y):
        y1, y2 = torch.chunk(y, dim=1, chunks=2)
        x1 = y1
        log_s, t = self.net(x1)
        x2 = (y2 - t) / torch.exp(log_s)
        x = torch.cat((x1, x2), 1)
        return x


class SimpleTransform(nn.Module):
    def __init__(self, dim, inflate_coef=2):
        super(SimpleTransform, self).__init__()
        internal_dim = int(dim * inflate_coef)
        self.model = nn.Sequential(
            nn.Linear(dim, internal_dim),
            nn.ReLU(),
            nn.Linear(internal_dim, 2 * dim),
        )
        nn.init.zeros_(self.model[-1].weight)
        nn.init.zeros_(self.model[-1].bias)


    def forward(self, x):
        out = self.model(x)
        log_s, t = torch.chunk(out, dim=1, chunks=2)
        return log_s, t


class MiniNormalizingFlow(nn.Module):
    def __init__(self, input_dim, steps=2, inflate_coef=2):
        super(MiniNormalizingFlow, self).__init__()
        self.input_dim = input_dim
        self.transforms = nn.ModuleList()
        for i in range(steps):
            self.transforms.append(ActNorm(input_dim))
            net = SimpleTransform(input_dim//2, inflate_coef)
            self.transforms.append(AffineCouplingLayer(net))
            if i != steps - 1:
                self.transforms.append(SwitchSides())

        self.transforms = nn.Sequential(*self.transforms)
        # self.loc = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer('loc', torch.zeros(input_dim))
        # self.log_scale = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer('log_scale', torch.zeros(input_dim))

    def forward(self, x, ret_z=False):
        z = x
        log_det = 0
        for t in self.transforms:
            z_, ld = t(z)
            if z_.isinf().any() or z_.isnan().any():
                breakpoint()
            z = z_
            log_det += ld
        dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))
        log_p_z = dist.log_prob(z).sum(1)
        log_p_x = (log_p_z + log_det) / self.input_dim
        if ret_z:
            return log_p_x, z
        return log_p_x

    def inverse(self, z):
        for t in reversed(self.transforms):
            z = t.inverse(z)
        return z

    def sample(self, num_samples):
        dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))
        z = dist.sample(torch.Size([num_samples]))
        return self.inverse(z)

    def log_prob(self, x, ret_z=False):
        return self.forward(x, ret_z=ret_z)
    
    
class MiniGlow(nn.Module):
    def __init__(self, input_dim=256, steps=2, inflate_coef=2, device=None):
        super(MiniGlow, self).__init__()
        self.input_dim = input_dim

        self.transforms = []
        for i in range(steps):
            self.transforms.append(ActNorm(input_dim))
            self.transforms.append(Conv1x1(input_dim, device))
            self.transforms.append(AffineCouplingLayer(SimpleTransform(input_dim//2, inflate_coef)))
            # if i != num_steps-1:
            #     self.transforms.append(SwitchSides())

        self.transforms = nn.Sequential(*self.transforms)

        self.register_buffer('loc', torch.zeros(input_dim))
        self.register_buffer('log_scale', torch.zeros(input_dim))

    def z_dist(self):
        z_dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))
        return z_dist

    def log_prob(self, x):
        z = x
        log_abs_det = 0.
        for m in self.transforms[1:]:
            z, ld_layer = m(z)
            log_abs_det += ld_layer
        log_pz = self.z_dist().log_prob(z).sum(-1)
        log_px = (log_pz + log_abs_det) / self.input_dim
        return log_px

    def forward(self, x):
        z = x
        for m in self.transforms:
            z, _ = m(z)
        return z

    def inverse(self, z):
        for m in reversed(self.transforms):
            z = m.inverse(z)
        return z

    def sample(self, num_samples, T=1):
        z_dist = torch.distributions.Normal(self.loc, torch.exp(self.log_scale))
        z = z_dist.sample(torch.Size([num_samples])) * T
        x = self.inverse(z)
        return x


class NormalizingFlowGMM(nn.Module):
    def __init__(self, input_dim, num_steps=2, n_components=19):
        super(NormalizingFlowGMM, self).__init__()
        self.input_dim = input_dim
        self.transforms = nn.ModuleList()
        self.n_components = n_components
        for i in range(num_steps):
            self.transforms.append(Conv1x1(input_dim))
            self.transforms.append(AffineCouplingLayer(SimpleTransform(input_dim//2, 2)))
            if i != num_steps-1:
                self.transforms.append(SwitchSides())
                self.transforms.append(ActNorm(input_dim))
        # for i in range(num_steps):
        #     self.transforms.append(AffineCouplingLayer(SimpleTransform(input_dim//2, 0.5)))
        #     if i != num_steps-1:
        #         self.transforms.append(SwitchSides())
        self.transforms = nn.Sequential(*self.transforms)
        # self.mu = nn.Parameter(torch.randn(1, n_components, input_dim), requires_grad=False)
        # self.register_buffer('mu',
        #                      10 * torch.nn.init.orthogonal_(torch.empty((n_components, input_dim))).expand(1, -1, -1))
        # self.mu = nn.Parameter(torch.eye(n_components, input_dim).unsqueeze(0), requires_grad=False)
        self.mu = nn.Parameter(10 * torch.nn.init.orthogonal_(torch.empty((n_components, input_dim))).expand(1, -1, -1), requires_grad=False)

        self.phi = nn.Parameter(torch.ones(n_components), requires_grad=False)
        self.register_buffer('loc', torch.zeros(input_dim))
        self.register_buffer('log_scale', torch.zeros(input_dim))

    def z_dist(self, loc):
        z_dist = torch.distributions.Normal(loc, torch.exp(self.log_scale))
        return z_dist

    def cond_log_prob(self, x, label, train=True, ret_p_z=False):
        z = x
        log_abs_det = 0.
        for m in self.transforms:
            z, ld_layer = m(z)
            log_abs_det += ld_layer
        mu = torch.zeros_like(x)
        for i in range(self.n_components):
            mu[label == i] = self.mu[:, i, :].squeeze()
        log_p_z_c = - (0.5 * (z-mu)**2 - math.log(math.sqrt(2*math.pi))).sum(-1)
        # log_pz_c = self.z_dist(self.mu[:, c, :]).log_prob(z).sum(-1)
        if train:
            log_p_x_c = (log_p_z_c + log_abs_det) / self.input_dim # train p_x
        else:
            log_p_x_c = log_p_z_c # eval p_z
        if ret_p_z:
            return log_p_x_c, log_p_z_c

        return log_p_x_c

    def log_prob(self, x, train=True, ret_z=False):
        z = x
        log_abs_det = 0.
        for m in self.transforms:
            z, ld_layer = m(z)
            log_abs_det += ld_layer
        out = []
        for i in range(self.n_components):
            l = i * torch.ones(x.shape[0], dtype=torch.int64).to(x.device)
            log_px = self.cond_log_prob(x, l, train)
            out.append(log_px)
        log_p_z_c = torch.stack(out).T
        if train:
            p_z_c = log_p_z_c.exp() #train
        else:
            p_z_c = log_p_z_c # eval
        s = - torch.sum(p_z_c, dim=1)
        if not ret_z:
            return p_z_c, s
        else:
            return p_z_c, s, z


class ResidualBlock(nn.Module):
    def __init__(self, dim, use_bn=False):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(dim, dim))
        if use_bn:
            self.layers.append(nn.BatchNorm1d(dim))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Linear(dim, dim))
        if use_bn:
            self.layers.append(nn.BatchNorm1d(dim))
        self.f_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.layers(x)
        out = out + identity
        return self.f_relu(out)


class ResNet(nn.Module):
    def __init__(self, in_dim, num_blocks=1):
        super(ResNet, self).__init__()
        self.dim = in_dim
        out_dim = 2 * in_dim
        mid_dim = 4 * in_dim
        layers = [nn.Linear(in_dim, mid_dim)] + \
                 [ResidualBlock(mid_dim) for _ in range(num_blocks)] + \
                 [nn.Linear(mid_dim, out_dim)]

        nn.init.zeros_(layers[-1].weight)
        if hasattr(layers[-1], 'bias'):
            nn.init.zeros_(layers[-1].bias)

        self.model = nn.Sequential(*layers)

    def forward(self, x, cond=None):
        out = self.model(x)
        log_s, t = torch.chunk(out, dim=1, chunks=2)
        return log_s, t


class NormalizingFlowGMMResnet(NormalizingFlowGMM):
    def __init__(self, input_dim, num_steps=2, n_components=10):
        super(NormalizingFlowGMM, self).__init__()
        self.input_dim = input_dim
        self.transforms = []
        self.n_components = n_components
        for i in range(num_steps):
            self.transforms.append(Conv1x1(input_dim))
            self.transforms.append(AffineCouplingLayer(ResNet(input_dim//2, 2)))
            if i != num_steps-1:
                self.transforms.append(SwitchSides())
                self.transforms.append(ActNorm(input_dim))

        self.transforms = nn.Sequential(*self.transforms)

        self.register_buffer('mu', 10*torch.nn.init.orthogonal_(torch.empty((n_components,input_dim))).expand(1, -1, -1))
        # self.mu = nn.Parameter(5*torch.nn.init.orthogonal_(torch.empty((n_components,input_dim))).expand(1, -1, -1), requires_grad=False)
        # self.mu = nn.Parameter(torch.eye(1, n_components, input_dim), requires_grad=False)
        self.phi = nn.Parameter(torch.ones(n_components), requires_grad=False)
        self.register_buffer('loc', torch.zeros(input_dim))
        self.register_buffer('log_scale', torch.zeros(input_dim))


def get_model_id(args):
    if args.model == "nflow":
        return 'mini-flow'
    elif args.model == "mglow":
        return 'mini-glow'
    elif args.model == "gmmflow":
        return 'gmmflow'


def get_model(args, data_shape):
    if args.model == "nflow":
        return MiniNormalizingFlow(input_dim=data_shape, steps=args.steps, inflate_coef=args.inflate_coef)
    elif args.model == "mglow":
        return MiniGlow(input_dim=data_shape, steps=args.steps, inflate_coef=args.inflate_coef, device=args.device)
    elif args.model in ["gmmflow", "mutualflow", "scratchflow", "frozenflow", "dualflow", "backdoorflow", "tripleflow", "nonegflow"]:
        return NormalizingFlowGMM(input_dim=data_shape, num_steps=args.steps, n_components=args.num_classes)
    elif args.model == "resnetmutualflow" or args.model == "resnettripleflow":
        return NormalizingFlowGMMResnet(input_dim=data_shape, num_steps=args.steps, n_components=args.num_classes)
