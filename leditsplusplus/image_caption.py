import torch
import numpy as np
import pandas as pd
import random
import argparse, os, json
from tqdm import tqdm
from PIL import Image
from utils import *
from constants import *
from pipeline_semantic_stable_diffusion_img2img_solver import SemanticStableDiffusionImg2ImgPipeline_DPMSolver
from torch import autocast, inference_mode
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from transformers import AutoProcessor, BlipForConditionalGeneration

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_image_dir', type=str, default='examples')
    parser.add_argument('--blip_model_id', type=str, default='Salesforce/blip-image-captioning-large')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    return parser.parse_args()

def randomize_seed_fn(seed, is_random):
    if is_random:
        seed = random.randint(0, np.iinfo(np.int32).max)
    return seed

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class BLIP:
    def load_model(self, blip_model_id, device):
        self.device = device
        self.blip_processor = AutoProcessor.from_pretrained(blip_model_id)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id, torch_dtype=torch.float16).to(device)

    def caption_image(self, input_image):
        inputs = self.blip_processor(images=input_image, return_tensors='pt').to(self.device, torch.float16)
        pixel_values = inputs.pixel_values

        generated_ids = self.blip_model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_ids, generated_caption
    


if __name__ == '__main__':
    args = get_args()
    blip = BLIP()
    blip.load_model(args.blip_model_id, args.device)

    datas = {}
    for csv in os.listdir('data/VisA/results'):
        df = pd.read_csv(os.path.join('data/VisA/results', csv))
        for d in tqdm(df.iterrows()):
            data = d[1]
            if data['label'] == 'normal':
                continue
            img_path = data['image']
            img = Image.open(os.path.join('data/VisA', img_path))
            generated_ids, generated_caption = blip.caption_image(img)
            # print(generated_caption)
            datas[img_path] = generated_caption

    with open('VisA_caption.json', 'w') as f:
        json.dump(datas, f, indent=2)

