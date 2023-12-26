import torch
import numpy as np
import random
import argparse, os, json
from tqdm import tqdm
from PIL import Image
from utils import *
from constants import *
from pipeline_semantic_stable_diffusion_img2img_solver import SemanticStableDiffusionImg2ImgPipeline_DPMSolver
from diffusers import StableDiffusionPipeline, AutoencoderKL
from diffusers.schedulers import DDIMScheduler
from scheduling_dpmsolver_multistep_inject import DPMSolverMultistepSchedulerInject
from transformers import AutoProcessor, BlipForConditionalGeneration

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_image_dir', type=str, default='examples')
    parser.add_argument('--input_prompt_path', type=str, default='examples/prompts.json')
    # parser.add_argument('--output_image_dir', type=str, default='outputs')
    # parser.add_argument('--sd_model_id', type=str, default='runwayml/stable-diffusion-v1-5') # alternative model: 'stabilityai/stable-diffusion-2-1-base'
    parser.add_argument('--sd_model_id', type=str, default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--vae_model_id', type=str, default='stabilityai/sd-vae-ft-mse')
    parser.add_argument('--blip_model_id', type=str, default='Salesforce/blip-image-captioning-base')
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


class LeditsPlusPlus:

    wts_tensor = None
    zs_tensor = None
    attention_store = None
    text_cross_attention_maps = ['']

    def __init__(self):
        pass

    def load_model(self, sd_model_id, vae_model_id, blip_model_id, device):
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=torch.float16)
        self.pipe = SemanticStableDiffusionImg2ImgPipeline_DPMSolver.from_pretrained(sd_model_id, vae=self.vae, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(device)
        self.pipe.scheduler = DPMSolverMultistepSchedulerInject.from_pretrained(sd_model_id, subfolder='scheduler', algorithm_type='sde-dpmsolver++', solver_order=2)
        self.blip_processor = AutoProcessor.from_pretrained(blip_model_id)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id, torch_dtype=torch.float16).to(device)

    def caption_image(self, input_image):
        inputs = self.blip_processor(images=input_image, return_tensors='pt').to(self.device, torch.float16)
        pixel_values = inputs.pixel_values

        generated_ids = self.blip_model.generate(pixel_values=pixel_values, max_length=50)
        generated_caption = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_ids, generated_caption
    
    def edit(self, args, prompt):
        self.parse_editing_args(prompt)

        mask_type = self.editing_args['mask_type']
        if(mask_type == 'No mask'):
            use_cross_attn_mask = False
            use_intersect_mask = False
        elif(mask_type=='Cross Attention Mask'):
            use_cross_attn_mask = True
            use_intersect_mask = False 
        elif(mask_type=='Intersect Mask'):
            use_cross_attn_mask = False
            use_intersect_mask = True 

        randomize_seed = self.editing_args['randomize_seed']
        seed = self.editing_args['seed']
        if randomize_seed:
            seed = randomize_seed_fn(seed, randomize_seed)
        seed_everything(seed)

        do_inversion = self.editing_args['do_inversion']
        input_image = self.editing_args['input_image']
        src_prompt = self.editing_args['src_prompt']
        src_cfg_scale = self.editing_args['src_cfg_scale']
        steps = self.editing_args['steps']
        skip = self.editing_args['skip']
        if do_inversion or randomize_seed:
            zs_tensor, wts_tensor = self.pipe.invert(
            image_path = input_image,
            source_prompt =src_prompt,
            source_guidance_scale= src_cfg_scale,
            num_inversion_steps = steps,
            skip = skip,
            eta = 1.0,
            )
            wts = wts_tensor
            zs = zs_tensor
            
            self.wts_tensor = wts
            self.zs_tensor = zs
            
            do_inversion = False
        
        image_caption = self.editing_args['image_caption']
        tar_prompt = self.editing_args['tar_prompt']
        if image_caption.lower() == tar_prompt.lower(): # if image caption was not changed, run pure sega
            tar_prompt = ''
        
        edit_concept_1 = self.editing_args['edit_concept_1']
        edit_concept_2 = self.editing_args['edit_concept_2']
        edit_concept_3 = self.editing_args['edit_concept_3']
        neg_guidance_1 = self.editing_args['neg_guidance_1']
        neg_guidance_2 = self.editing_args['neg_guidance_2']
        neg_guidance_3 = self.editing_args['neg_guidance_3']
        warmup_1 = self.editing_args['warmup_1']
        warmup_2 = self.editing_args['warmup_2']
        warmup_3 = self.editing_args['warmup_3']
        guidnace_scale_1 = self.editing_args['guidnace_scale_1']
        guidnace_scale_2 = self.editing_args['guidnace_scale_2']
        guidnace_scale_3 = self.editing_args['guidnace_scale_3']
        threshold_1 = self.editing_args['threshold_1']
        threshold_2 = self.editing_args['threshold_2']
        threshold_3 = self.editing_args['threshold_3']
        editing_args = {}
        if edit_concept_1 != '' or edit_concept_2 != '' or edit_concept_3 != '':
            editing_args = dict(
            editing_prompt = [edit_concept_1,edit_concept_2,edit_concept_3],
            reverse_editing_direction = [ neg_guidance_1, neg_guidance_2, neg_guidance_3,],
            edit_warmup_steps=[warmup_1, warmup_2, warmup_3,],
            edit_guidance_scale=[guidnace_scale_1,guidnace_scale_2,guidnace_scale_3],
            edit_threshold=[threshold_1, threshold_2, threshold_3],
            edit_momentum_scale=0,
            edit_mom_beta=0,
            eta=1,
            use_cross_attn_mask=use_cross_attn_mask,
            use_intersect_mask=use_intersect_mask
            )

        tar_cfg_scale = self.editing_args['tar_cfg_scale']
        latnets = wts[-1].expand(1, -1, -1, -1)
        
        attention_store = self.editing_args['attention_store']
        text_cross_attention_maps = self.editing_args['text_cross_attention_maps']
        sega_out, attention_store, text_cross_attention_maps = self.pipe(prompt=tar_prompt, 
                            init_latents=latnets, 
                            guidance_scale = tar_cfg_scale,
                            # num_images_per_prompt=1,
                            # num_inference_steps=steps,
                            # use_ddpm=True,  
                            # wts=wts.value, 
                            zs=zs, attention_store=attention_store, text_cross_attention_maps=text_cross_attention_maps, **editing_args)
        
        self.attention_store = attention_store
        self.text_cross_attention_maps = text_cross_attention_maps

        return sega_out.images[0]
    
    def parse_editing_args(self, prompt):
        results = {}
        image = Image.open(prompt['image_path'])
        
        out_path = prompt['output_image_path']
        out_dir = os.path.dirname(out_path)
        label_dir = os.path.join(out_dir, 'label')
        os.makedirs(label_dir, exist_ok=True)
        out_image_name = out_path.split('/')[-1]
        image.save(os.path.join(label_dir, out_image_name))

        results['input_image'] = image
        results['wts'] = self.wts_tensor
        results['zs'] = self.zs_tensor
        results['attention_store'] = self.attention_store
        results['text_cross_attention_maps'] = None
        generated_ids, generated_caption = self.caption_image(image)

        results['tar_prompt'] = generated_caption if 'tar_prompt' not in prompt else prompt['tar_prompt']
        results['image_caption'] = generated_caption
        results['steps'] = 50 if 'steps' not in prompt else prompt['steps']
        results['skip'] = 25 if 'skip' not in prompt else prompt['skip']
        results['tar_cfg_scale'] = 7.5 if 'tar_cfg_scale' not in prompt else prompt['tar_cfg_scale']
        results['edit_concept_1'] = '' if 'edit_concept_1' not in prompt else prompt['edit_concept_1']
        results['edit_concept_2'] = '' if 'edit_concept_2' not in prompt else prompt['edit_concept_2']
        results['edit_concept_3'] = '' if 'edit_concept_3' not in prompt else prompt['edit_concept_3']
        results['guidnace_scale_1'] = 7 if 'guidnace_scale_1' not in prompt else prompt['guidnace_scale_1']
        results['guidnace_scale_2'] = 7 if 'guidnace_scale_2' not in prompt else prompt['guidnace_scale_2']
        results['guidnace_scale_3'] = 7 if 'guidnace_scale_3' not in prompt else prompt['guidnace_scale_3']
        results['warmup_1'] = 2 if 'warmup_1' not in prompt else prompt['warmup_1']
        results['warmup_2'] = 2 if 'warmup_2' not in prompt else prompt['warmup_2']
        results['warmup_3'] = 2 if 'warmup_3' not in prompt else prompt['warmup_3']
        results['neg_guidance_1'] = False if 'neg_guidance_1' not in prompt else prompt['neg_guidance_1']
        results['neg_guidance_2'] = False if 'neg_guidance_2' not in prompt else prompt['neg_guidance_2']
        results['neg_guidance_3'] = False if 'neg_guidance_3' not in prompt else prompt['neg_guidance_3']
        results['threshold_1'] = 0.95 if 'threshold_1' not in prompt else prompt['threshold_1']
        results['threshold_2'] = 0.95 if 'threshold_2' not in prompt else prompt['threshold_2']
        results['threshold_3'] = 0.95 if 'threshold_3' not in prompt else prompt['threshold_3']
        results['do_reconstruction'] = True
        results['reconstruction'] = None
        results['do_inversion'] = True
        results['seed'] = 540000000 if 'seed' not in prompt else prompt['seed']
        results['randomize_seed'] = False if 'randomize_seed' not in prompt else prompt['randomize_seed']
        results['src_prompt'] = '' if 'src_prompt' not in prompt else prompt['src_prompt']
        results['src_cfg_scale'] = 3.5 if 'src_cfg_scale' not in prompt else prompt['src_cfg_scale']
        results['mask_type'] = 'Intersect Mask'
        self.editing_args = results
    


if __name__ == '__main__':
    args = get_args()

    leditsplusplus = LeditsPlusPlus()
    leditsplusplus.load_model(args.sd_model_id, args.vae_model_id, args.blip_model_id, args.device)
    
    with open(args.input_prompt_path, "r") as file:
        input_prompts = json.load(file)
    for prompt in tqdm(input_prompts):
        output_image= leditsplusplus.edit(args, prompt)
        output_image_dir = os.path.dirname(prompt['output_image_path'])
        os.makedirs(output_image_dir, exist_ok=True)
        output_image.save(prompt['output_image_path'])
