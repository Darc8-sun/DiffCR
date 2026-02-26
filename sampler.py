import torch
from ldm.modules.diffusionmodules.util import extract_into_tensor,make_beta_schedule
import numpy as np
from typing import  Dict, Optional, Tuple
def extract_into_tensor2(a, t, x_shape):
    if a.dim()-1:
        t=t.unsqueeze(dim=1)
    b, *_ = t.shape
    out = a.gather(0, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)






class DDIMSolver:
    def __init__(self, alpha_cumprods, timesteps=1000, ddim_timesteps=50):
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0, pred_noise, timestep_index):
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev.float()



class CRSampler:
    def __init__(
            self,
            model: "DiffCR",
            schedule: str = "linear",
            var_type: str = "fixed_small",
            ddim_samplestep: int=50,
    ) -> "CRSampler":
        self.model = model
        self.original_num_steps = model.num_timesteps
        self.schedule = schedule
        self.var_type = var_type
        self.ddim_samplestep=ddim_samplestep

    def make_schedule(self, num_steps: int,start_t_inds=50) -> None:
        betas = make_beta_schedule(
            self.schedule, self.original_num_steps, linear_start=self.model.linear_start,
            linear_end=self.model.linear_end
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        step_ratio = self.original_num_steps  // self.ddim_samplestep
        self.ddim_timesteps = (np.arange(1,  self.ddim_samplestep + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alphas_cumprod[self.ddim_timesteps]
        self.ddim_timesteps_prev =np.asarray(
            [0] + self.ddim_timesteps[:-1].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_timesteps_prev = torch.from_numpy(self.ddim_timesteps_prev).long()
        if isinstance(start_t_inds,np.ndarray) and len(start_t_inds)!=1:
            timesteps=[]
            ddim_alpha_cumprods_prev=[]
            for start_t_ind in start_t_inds:
                indices = torch.tensor(sorted(list(space_timesteps(int(start_t_ind), str(num_steps+1)))))
                timestep=self.ddim_timesteps[indices[1:]].numpy()
                timesteps.append(np.expand_dims(timestep,axis=1))
                dacp =np.asarray(alphas_cumprod[np.asarray(
                    [0] + timestep[:-1].tolist()
                )])
                ddim_alpha_cumprods_prev.append(torch.from_numpy(dacp).unsqueeze(1))
            self.timesteps=np.concatenate(timesteps,axis=1)
            self.ddim_alpha_cumprods_prev = torch.cat(ddim_alpha_cumprods_prev, dim=1)
        else:
            start_t_inds=int(start_t_inds)
            indices = torch.tensor(sorted(list(space_timesteps(int(start_t_inds), str(num_steps + 1)))))
            self.timesteps = self.ddim_timesteps[indices[1:]].numpy()
            self.ind = indices
            # self.timesteps = self.ddim_timesteps[indices].numpy()
            # self.ddim_alpha_cumprods_prev =np.asarray(alphas_cumprod[np.asarray(
            #     [0] + self.timesteps[:-1].tolist()
            # )])
            # self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)
            self.ddim_alpha_cumprods_prev = np.asarray(alphas_cumprod[np.asarray(
                [0] + self.timesteps[:-1].tolist()
            )])
            self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)
    def ddim_sample(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        alpha_cumprod_prev = extract_into_tensor2(self.ddim_alpha_cumprods_prev.to(x_start.device), t, x_start.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * noise
        x_prev = alpha_cumprod_prev.sqrt() * x_start + dir_xt
        if t[0]==0:
            return x_start
        else:
            return x_prev.float()

    def p_sample(
            self,
            x: torch.Tensor,
            cond: Dict[str, torch.Tensor],
            t: torch.Tensor,
            index:torch.Tensor,
            cfg_scale: float,
            uncond: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.:
            # with torch.no_grad():
            model_output = self.model.apply_model(x, t, cond)
            eps= self.model.predict_x0(x, t, model_output)
            e_t = self.model.tar(eps, cond['decompress_result'][0], t)
            x_0_P = self.model.build_cm_function(x,e_t,t)
        else:
            raise NotImplementedError
        pred=self.ddim_sample(x_0_P,index,model_output)
        return pred,e_t
    def sample_latent(
            self,
            steps: int,
            shape: Tuple[int],
            cond_img: torch.Tensor,
            x_T: Optional[torch.Tensor] = None,
            cfg_scale: float = 1.,
            semantic_latent=None,
            start_t_ind=50,
            prompt=None,
    ) -> torch.Tensor:
        if isinstance(start_t_ind,np.ndarray):
            start_t_ind=np.where(start_t_ind<steps+1,steps+1,start_t_ind)

        else:
            start_t_ind = int(start_t_ind)
            start_t_ind = max(start_t_ind, steps)
        self.make_schedule(num_steps=steps,start_t_inds=start_t_ind)
        device = next(self.model.parameters()).device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T
        try:
            if start_t_ind!=50:
                img=cond_img
            else:
                pass
        except:
            img = cond_img
        time_range = np.flipud(self.timesteps)  # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        if semantic_latent is not None:
            cond = {
                "c_concat": [cond_img],
                "c_crossattn": [semantic_latent],
                'decompress_result':[cond_img],
            }
        else:
            cond = {
                "c_concat": [cond_img],
                "c_crossattn": [self.model.get_learned_conditioning(prompt, dad=True)],
                'decompress_result':[cond_img],
            }

        for i, step in enumerate(time_range):  # iterator
            try:
                ts = torch.full((b,), step, device=device, dtype=torch.long)
            except:
                ts = torch.tensor(step, device=device,dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            if i==0:
                img=self.model.input_scaler(img,ts)
            img,_ = self.p_sample(
                img, cond, ts, index=index,cfg_scale=cfg_scale, uncond=None)
        return img