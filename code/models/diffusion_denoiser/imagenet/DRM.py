import torch
import torch.nn as nn 
import timm
import os
from transformers import ViTImageProcessor, ViTForImageClassification
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from models.diffusion_denoiser.imagenet.guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)


class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


class DiffusionRobustModel(nn.Module):
    def __init__(self, classifier_name="beit",device="cuda:0"):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        mdpath = f"{os.path.dirname(__file__)}/256x256_diffusion_uncond.pt"
        model.load_state_dict(
            torch.load(mdpath,map_location=device,weights_only=False)
        )
        warnings.warn(f"\033[7m Using trained model from {mdpath} \033[0m",UserWarning)
        model.eval().cuda(device)

        self.model = model 
        self.diffusion = diffusion 
        self.device=device

        # Load the BEiT model
        # classifier = timm.create_model('beit_large_patch16_512', pretrained=True)
        classifier = ViTForImageClassification.from_pretrained(f"{os.path.dirname(__file__)}/../vit-patch16-224")
        # classifier = timm.create_model('beit_base_patch16_224.in22k_ft_in22k_in1k', pretrained=False, checkpoint_path = "./beit_base_patch16_224.in22k_ft_in22k_in1k/pytorch_model.bin")
        # classifier = timm.create_model('beit_large_patch16_512.in22k_ft_in22k_in1k', pretrained=False, checkpoint_path = "./timm_beit_large_patch16_512/pytorch_model.bin")
        # classifier = timm.create_model("vit_base_patch16_224.dino", pretrained=False,checkpoint_path = "./vit_base_patch16_224.dino/pytorch_model.bin" )
        classifier.eval().cuda(self.device)

        self.classifier = classifier
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
        device_ids = [0]
        # self.model = torch.nn.DataParallel(self.model, device_ids =device_ids).cuda()
        # self.diffusion  = torch.nn.DataParallel(self.diffusion, device_ids =device_ids).cuda()
        # self.classifier = torch.nn.DataParallel(self.classifier, device_ids =device_ids).cuda()

    def forward(self, x, t):
        x_in = x * 2 -1
        imgs = self.denoise(x_in, t)

        imgs = torch.nn.functional.interpolate(imgs, (224, 224), mode='bicubic', antialias=True)

        imgs = torch.tensor(imgs).cuda(self.device)
        with torch.no_grad():
            out = self.classifier(imgs)

        return out

    def denoise(self, x_start, t, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).cuda(self.device)

        noise = torch.randn_like(x_start)
        
        #:param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).cuda(self.device)
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out