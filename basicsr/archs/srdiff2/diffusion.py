from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from basicsr.archs.srdiff.module_util import default
from basicsr.archs.srdiff.sr_utils import SSIM, PerceptualLoss
from basicsr.archs.restormer.hparams import hparams
from collections import namedtuple
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, rrdb_net, timesteps=1000, sampling_timesteps=100,loss_type='l1'):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.rrdb = rrdb_net
        # condition net
        self.ssim_loss = SSIM(window_size=11)

        if hparams['beta_schedule'] == 'cosine':
            betas = cosine_beta_schedule(timesteps, s=hparams['beta_s'])
        if hparams['beta_schedule'] == 'linear':
            betas = get_beta_schedule(timesteps, beta_end=hparams['beta_end'])

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # for ddim
        self.sampling_timesteps =sampling_timesteps
        self.ddim_sampling_eta = 0.0

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
        self.sample_tqdm = True

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, noise_pred, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    def forward(self, img_hr, img_lr, t=None, *args, **kwargs):
        x = img_hr
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long() \
            if t is None else torch.LongTensor([t]).repeat(b).to(device)

        _, cond = self.rrdb(img_lr, True)

        # p_losses, x_tp1, noise_pred, x_t, x_t_gt, x_0
        p_losses, x_tp1, noise_pred, x_t, x_t_gt, x_0 = self.p_losses(x, t, cond, *args, **kwargs)

        return p_losses

    def p_losses(self, x_start, t, cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_tp1_gt = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_t_gt = self.q_sample(x_start=x_start, t=t - 1, noise=noise)
        noise_pred = self.denoise_fn(x_tp1_gt, t, cond)
        x_t_pred, x0_pred = self.p_sample(x_tp1_gt, t, cond, noise_pred=noise_pred)

        if self.loss_type == 'l1':
            loss = (noise - noise_pred).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        elif self.loss_type == 'ssim':
            loss = (noise - noise_pred).abs().mean()
            loss = loss + (1 - self.ssim_loss(noise, noise_pred))
        else:
            raise NotImplementedError()
        return loss, x_tp1_gt, noise_pred, x_t_pred, x_t_gt, x0_pred

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        t_cond = (t[:, None, None, None] >= 0).float()
        t = t.clamp_min(0)
        return (
                       extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                       extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
               ) * t_cond + x_start * (1 - t_cond)

    @torch.no_grad()
    def p_sample(self, x, t, cond, noise_pred=None, clip_denoised=True, repeat_noise=False):
        if noise_pred is None:
            noise_pred = self.denoise_fn(x, t, cond=cond)
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x0_pred = self.p_mean_variance(
            x=x, t=t, noise_pred=noise_pred, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0_pred

    @torch.no_grad()
    def sample(self, img_lr, shape, save_intermediate=False):
        device = self.betas.device
        b = shape[0]
        # 随机HR噪声作为输入
        img = torch.randn(shape, device=device)
        # LR作为条件
        _, cond = self.rrdb(img_lr, True)

        it = reversed(range(0, self.num_timesteps))
        if self.sample_tqdm:
            it = tqdm(it, desc='sampling loop time step', total=self.num_timesteps)
        images = []
        for i in it:
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond)
            if save_intermediate:
                images.append((img.cpu(), x_recon.cpu()))
        if save_intermediate:
            return img, images
        else:
            return img

    @torch.no_grad()
    def ddim_sample(self, x_in, shape):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        b = x_in.shape[0]
        total_timesteps, sampling_timesteps, eta = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        # imgs保存每一次采样结果
        imgs = [img]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((b,), time, device = device, dtype = torch.long)
            # self_cond = x_start if self.conditional else None
            self_cond = x_in
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            imgs.append(img)
        ret = img
        # ret = self.unnormalize(ret)
        return ret

    # add for ddim
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False,rederive_pred_noise = False):
        x_start = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn(x, t, x_self_cond))
        x_start.clamp_(-1., 1.)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start)


    @torch.no_grad()
    def interpolate(self, x1, x2, img_lr, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        cond = img_lr

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img, x_recon = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond)
        return img