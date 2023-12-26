import torch
import numpy as np
import argparse, os
import clip
# from transformers import CLIPFeatureExtractor, CLIPProcessor, CLIPModel
# from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from PIL import Image
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    # parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # type_name = 'chewinggum'
    # print(type_name)
    # parser.add_argument('--image_dir1', type=str, default=f'outputs/VisA_GPT_caption/Normal/{type_name}')
    # parser.add_argument('--image_dir2', type=str, default=f'outputs/VisA_GPT_caption/Normal/{type_name}/label')

    parser.add_argument('--base_image_dir', type=str, default='outputs/VisA_GPT/Normal')

    args = parser.parse_args()
    return args


# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    return preprocess(image).unsqueeze(0).to(device)

# Function to get the image embedding
def get_image_embedding(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

# Function to calculate cosine similarity
def cosine_similarity(image_path1, image_path2):
    embedding1 = get_image_embedding(image_path1)
    embedding2 = get_image_embedding(image_path2)
    cos_sim = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cos_sim.item()

def main(args):
    # Load the CLIP model
    # Example usage

    cos_sims = np.array([])
    types = ['candle', 'capsules', 'cashew', 'chewinggum']
    for type in tqdm(types):
        curr_dir = os.path.join(args.base_image_dir, type)
        for img_name in os.listdir(curr_dir):
            if not img_name.endswith('.JPG'):
                continue
            image_path1 = os.path.join(curr_dir, img_name)
            image_path2 = os.path.join(curr_dir, 'label', img_name)
            similarity = cosine_similarity(image_path1, image_path2)
            cos_sims = np.append(cos_sims, similarity)
    print(f"Cosine Similarity: {np.mean(cos_sims)}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    args = get_args()
    main(args)