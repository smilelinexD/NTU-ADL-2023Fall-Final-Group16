import numpy as np
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
import argparse, os
import json
from tqdm import tqdm
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--prompt', type=str, default='examples/VisA_GPT_caption.json')

    args = parser.parse_args()
    return args

def main(args):
    # Initialize the model
    metric = CLIPScore(model_name_or_path=args.model_name).to(args.device)

    with open(args.prompt, 'r') as f:
        prompts = json.load(f)
    
    scores = np.array([])
    for prompt in tqdm(prompts):
        image = Image.open(prompt['image_path'])
        image = (np.asarray(image)*255).astype(np.uint8)
        text = prompt['tar_prompt']
        score = metric(torch.from_numpy(image).to(args.device), text).detach().cpu().numpy()
        scores = np.append(scores, score)

    print(scores)
    print('CLIP score: ', scores.mean())

if __name__ == '__main__':
    args = get_args()
    main(args)