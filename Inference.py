import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from omegaconf import OmegaConf
from utils_l import instantiate_from_config, load_state_dict
from argparse import ArgumentParser
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

parser = ArgumentParser()
parser.add_argument("--model_dir",
                    default="./ckpt/CR_model/high_bit_01",
                    type=str)
parser.add_argument("--base_diffusion_ckpt",
                    default="./ckpt/SD21/v2-1_512-ema-pruned.ckpt",
                    type=str)
parser.add_argument("--img_path",default="./test_pic/anime.png",type=str)
parser.add_argument("--prompt",default='A close-up of an anime girl with wide eyes, blushing cheeks, and an open mouth, expressing intense shock or excitement.',type=str)
parser.add_argument("--config_dir", default="./Configs/diffcr_model.yaml", type=str)
parser.add_argument("--output_dir", default='./', type=str)
parser.add_argument("--device", default='cuda', type=str)
args = parser.parse_args()

def pad_to_multiple_of_64(image):
    c, h, w = image.shape

    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    pad_transform = transforms.Pad((0, 0, pad_w, pad_h))  # (left, top, right, bottom)
    padded_image = pad_transform(image)

    return padded_image,pad_w,pad_h
def back_to_org(image,pad_w,pad_h):
    h,w,c = image.shape
    return image[:h-pad_h,:w-pad_w,:]
def main():
    ###load model
    model_configs = OmegaConf.load(args.config_dir)
    model = instantiate_from_config(model_configs)
    load_state_dict(model, torch.load(
        args.base_diffusion_ckpt, map_location="cpu"), strict=False)
    load_state_dict(model, torch.load(
        args.model_dir, map_location="cpu"), load_compressor=True,
                    strict=False)
    model.to(args.device)
    model.freeze()


    ###load image
    totensor = ToTensor()
    img = totensor(Image.open(args.img_path).convert('RGB')).to(args.device)
    img, w, h = pad_to_multiple_of_64(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        compress_pic,bpp=model.inference_pipe((img-0.5)*2,txt=args.prompt,normlize=True,bits=8)
        compress_pic = back_to_org(compress_pic, w, h)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)



    ###save image
    save_name = os.path.join(args.output_dir, f'compress_image_bpp_{bpp:.4f}.png')
    plt.imsave(save_name,compress_pic)

if __name__ == "__main__":
   main()