import torch.nn.functional
from compressor import Get_compressor
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.util import extract_into_tensor,timestep_embedding
from ldm.modules.distributions.distributions import \
    DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from module import (Time_aware_refinementX,append_dims,scalings_for_boundary_conditions,ProjLayer,adaptive_instance_normalization)
from sampler import DDIMSolver, CRSampler
import math
from omegaconf import OmegaConf
import numpy as np
comp_configs=OmegaConf.load("./Configs/compressor/compX.yaml")

class ControlledUnetModel(UNetModel):
    def __init__(self,**kwargs):
        super(ControlledUnetModel, self).__init__(**kwargs)
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, ori=None, **kwargs):
        hs = []
        # with torch.no_grad():
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        if control is not None:
            h += control.pop()
        for i, module in enumerate(self.output_blocks):
            hs_ = hs.pop()
            # if h.shape[1] == 1280:
            #     hs_ = Fourier_filter(hs_, threshold=1, scale=self.s1)
            # if h.shape[1] == 640:
            #     hs_ = Fourier_filter(hs_, threshold=1, scale=self.s2)
            if only_mid_control or control is None:
                h = torch.cat([h, hs_], dim=1)
            else:
                h = torch.cat([h, hs_ + control.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        return self.out(h)
class DiffCR(LatentDiffusion):
    def __init__(
            self,
            control_key: str,
            samplesteps: int = 2,
            *args,
            **kwargs
    ) -> "DiffCR":
        super().__init__(*args, **kwargs)
        self.ddim_sample = DDIMSolver(
            self.alphas_cumprod.numpy(),
            timesteps=self.num_timesteps,
            ddim_timesteps=50, )
        self.sample_steps = samplesteps
        self.tar = Time_aware_refinementX(in_ch=4, out_ch=4, model_channels=320)
        self.topk = self.num_timesteps // 50
        self.count = 0
        self.ema_decay = 0.95
        self.sampler = CRSampler(self)
        self.noisequant = False
        self.control_key = control_key
        ddim_t = self.ddim_sample.ddim_timesteps
        self.snr_select, self.snr_index = self.SNR_diff[ddim_t].sort()
        self.base_txt_context = ['A high-resolution, 8K, ultra-realistic image, best quality, and extremely detail']
        self.control_scales = [1.0] * 13

        self.proj_layer = ProjLayer(
            in_dim=1024, out_dim=1024, hidden_dim=2048, drop_p=0.1, eps=1e-12)


    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x, return_fea=True)

    def input_scaler(self, z, t):
        return extract_into_tensor(self.sqrt_alphas_cumprod, t, z.shape) * z



    def Get_index_fromSNR(self, pred, org):
        # snr_d = torch.square(org).sum([1, 2, 3]) / torch.square(pred - org).sum([1, 2, 3])
        snr_d = 1.0 / torch.square(pred - org).mean([1, 2, 3])
        index_n = torch.bucketize(snr_d.detach().cpu(), self.snr_select)
        index_n = torch.where(index_n > 49, 49, index_n)
        del snr_d
        return self.snr_index[index_n].numpy()


    def get_learned_conditioning(self, c, dad=False):
        with torch.no_grad():
            assert len(c) == 2
            ctx = self.cond_stage_model.image_embedding(c[0]).float()
            ctx_txt = super().get_learned_conditioning(c[1])
            base_ctx = super().get_learned_conditioning(self.base_txt_context * ctx.shape[0])
        ctx = torch.cat([ctx.unsqueeze(dim=1), ctx_txt, base_ctx], dim=1)
        ctx = self.proj_layer(ctx)

        return ctx


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        # global active_task
        # active_task = t
        # self.apply(_change_task)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, only_mid_control=True)
        return eps


    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)


    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        compress_model = Get_compressor(comp_configs)
        self.compress_model = compress_model.eval()
        self.first_stage_model = model.eval()
        for param in self.first_stage_model.parameters():
            param.requires_grad = False


    @torch.no_grad()
    def sample_log(self, cond, steps, txt, start_t=50, x_T=None):
        samples = (self.decode_first_stage(self.sample_latent(cond, steps, txt, start_t)) + 1) / 2
        return samples

    @torch.no_grad()
    def Get_compress_result(self, img, txt):
        h, w = img.size(2), img.size(3)
        num_pixels = 1 * h * w
        c_enc = self.first_stage_model.encode(img, return_fea=True)
        compress_dict = self.compress_model(c_enc[1], c_enc[-1], noisequant=False)
        z = self.get_first_stage_encoding(c_enc[0], sample=False)
        c_hat = compress_dict['x_hat']
        bpp = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in compress_dict["likelihoods"].values()
        )
        ultra_bits=int(53)
        bpp=bpp+(ultra_bits/num_pixels)
        c_hat = self.get_first_stage_encoding(c_hat, sample=False)
        x_hat = (self.decode_first_stage(c_hat).detach() + 1.0) / 2.0
        ctx = [x_hat, txt]
        ctx = self.get_learned_conditioning(ctx)
        return dict(img_control=c_hat, x_hat=x_hat, bpp=bpp, ctx=ctx, z=z)
    def inference_pipe(self,img,txt=None,normlize=False,bits=8):
        txt =self.base_txt_context[0] if txt==None else txt
        comp_dict = self.Get_compress_result(img, txt)
        t_ind = self.Get_index_fromSNR(comp_dict["img_control"], comp_dict['z'])
        samples = self.sample_log(
            cond={"c_concat": [comp_dict["img_control"]], "c_crossattn": [comp_dict['ctx']]},
            steps=self.sample_steps,
            txt=txt,
            start_t=t_ind,
        )
        if normlize:
            samples = adaptive_instance_normalization(samples, img / 2.0 + 0.5,bits)
        samples = torch.clamp(samples, 0, 1)
        samples = np.round(np.transpose((samples[0]).clamp_(0, 1).cpu().detach().numpy() * 255, (1, 2, 0))).astype(
            np.uint8)
        return samples, comp_dict['bpp']
    def sample_latent(self, cond, steps, txt, start_t=50, x_stage=False):
        b, c, h, w = cond["c_concat"][0].shape
        shape = (b, self.channels, h, w)
        cond_img = cond["c_concat"][0]
        if len(cond["c_concat"]) == 1:
            cond["c_concat"].append(None)
        samples = self.sampler.sample_latent(
            steps, shape, cond_img,  start_t_ind=start_t,
            semantic_latent=cond["c_crossattn"][0])
        return samples

    def build_cm_function(self, x_noisy, x_pred, t):
        c_skip, c_out = scalings_for_boundary_conditions(t)
        c_skip, c_out = [append_dims(x, x_noisy.ndim) for x in [c_skip, c_out]]
        return c_skip * x_noisy + c_out * x_pred

    def predict_x0(self, x_t, t, eps):
        x0_hat = (x_t - (extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps)) / (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape))
        return x0_hat


    def get_first_stage_encoding(self, encoder_posterior, sample=True):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            if sample:
                z = encoder_posterior.sample()
            else:
                z = encoder_posterior.mode()
            # z = encoder_poterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z