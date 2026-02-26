import sys
from PIL import Image

if len(sys.argv) < 2:
  sys.exit("""Usage: python run.py path-to-image [path-to-image-2 ...]
Passing multiple images will optimize a single prompt across all passed images, useful for style transfer.
""")

config_path = "./PI/sample_config.json"

image_paths = sys.argv[1:]

# load the target image
images = [Image.open(image_path) for image_path in image_paths]

# defer loading other stuff until we confirm the images loaded
import argparse
import PI.open_clip as open_clip
from PI.optim_utils import *

print("Initializing...")
from main_elic import Makedataloader
from omegaconf import OmegaConf
device = "cuda"
image_size = 384

config=OmegaConf.load("configs/elic_latent.yaml")
config['dataset'].picsize=384
_,test_dataloader=Makedataloader(config,datarange=15000,crop='resize')

t=test_dataloader.dataset.__getitem__(0).reshape([1,3,384,384])

# load args
args = argparse.Namespace()
args.__dict__.update(read_json(config_path))

# You may modify the hyperparamters here
args.print_new_best = True

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L/14', pretrained="/data/xiayicong/checkpoint/vit_l14/open_clip_pytorch_model.bin", device=device)

print(f"Running for {args.iter} steps.")
if getattr(args, 'print_new_best', False) and args.print_step is not None:
  print(f"Intermediate results will be printed every {args.print_step} steps.")


path=test_dataloader.dataset.img_path[0]['path']
images = [Image.open(path.replace('HR/train',"/data/xiayicong/dataset/LSDIR/train"))]
# optimize prompt
import time
time_s=time.time()
learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=images)
time_e=time.time()-time_s
print(time_e,learned_prompt)
