import os
import argparse
import random
import time
import numpy as np

import PIL
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
from accelerate import Accelerator
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import \
	StableDiffusionSafetyChecker
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (PIL_INTERPOLATION, deprecate,
							 is_accelerate_available, is_accelerate_version,
							 logging, randn_tensor)
'''

from transformers import (AutoProcessor, CLIPFeatureExtractor,
						  CLIPImageProcessor, CLIPModel, CLIPTextModel,
						  CLIPTokenizer, CLIPVisionModel)

from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import (dot_score, normalize_embeddings,
										semantic_search)





def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--max_token', type=int, default = 3)
	parser.add_argument('--before', type = str, default = "./images/painting1/0_0.png")
	parser.add_argument('--after', type = str, default = "./images/painting1/0_1.png")
	parser.add_argument('--model_id', type = str, default = "openai/clip-vit-large-patch14")
	parser.add_argument('--device', type = str, default = "mps")
	return parser.parse_args()








def text_projection( clip_model, processor, before_images, after_images, device='mps'):
	with torch.no_grad():
			
		before_input = processor( images=before_images, return_tensors="pt").to(device)
		pooled_features_before = clip_model.vision_model(before_input['pixel_values'])[1]
		pooled_features_before = clip_model.visual_projection(pooled_features_before)

		after_input = processor( images=after_images, return_tensors="pt").to(device)
		pooled_features_after = clip_model.vision_model(after_input['pixel_values'])[1]
		pooled_features_after = clip_model.visual_projection(pooled_features_after)

		edit_direction_embed = pooled_features_after.mean(dim=0) - pooled_features_before.mean(dim=0)
		edit_direction_embed = edit_direction_embed / edit_direction_embed.norm(p=2, dim=-1, keepdim=True)

		return edit_direction_embed


## i don't know why stable diff. need this
def preprocess(image):
	if isinstance(image, torch.Tensor):
		return image
	elif isinstance(image, PIL.Image.Image):
		image = [image]

	if isinstance(image[0], PIL.Image.Image):
		w, h = image[0].size
		w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

		image = [np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image]
		image = np.concatenate(image, axis=0)
		image = np.array(image).astype(np.float32) / 255.0
		image = image.transpose(0, 3, 1, 2)
		image = 2.0 * image - 1.0
		image = torch.from_numpy(image)
	elif isinstance(image[0], torch.Tensor):
		image = torch.cat(image, dim=0)
	return image



def main(args):
	clip_model = CLIPModel.from_pretrained(args.model_id)
	processor = AutoProcessor.from_pretrained(args.model_id)
	tokenizer = processor.tokenizer
	clip_model.eval().to(args.device)
	
	before_image = Image.open(args.before).resize((512, 512)).convert('RGB')
	after_image = Image.open(args.after).resize((512, 512)).convert('RGB')


	clip_embedding = text_projection(clip_model, processor, before_image, after_image).unsqueeze(0).to("cpu")
	#print(clip_embedding, clip_embedding.shape)
	if args.max_token > 1:
		clip_embedding = torch.cat([clip_embedding] * args.max_token, 0)
	#print(clip_embedding, clip_embedding.shape)
	
	curr_embeds = normalize_embeddings(clip_embedding).to(args.device)

	embedding_matrix = clip_model.text_model.embeddings.token_embedding.weight

	with torch.no_grad():
		for i in range( embedding_matrix.shape[0]):
			#print( tokenizer(tokenizer.decode(torch.tensor([i+1000]))) )
			#time.sleep(1)
			if ( tokenizer(tokenizer.decode(torch.tensor([i])))['input_ids'][1] != i ):
				embedding_matrix[i]=torch.zeros_like(embedding_matrix[i])
		
	embedding_matrix = embedding_matrix.detach()

	embedding_matrix = normalize_embeddings(embedding_matrix).to(args.device)


	with torch.no_grad():
		print(curr_embeds.shape, embedding_matrix.shape)
		hits = semantic_search(curr_embeds, embedding_matrix, 
			query_chunk_size=curr_embeds.shape[0], 
			top_k=1,
			score_function=dot_score)
		print(hits)

		nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device).unsqueeze(0)
		print(nn_indices)

		texts = []
		input_ids = nn_indices.detach().cpu().numpy()
		for input_ids_i in input_ids:
			texts.append(tokenizer.decode(input_ids_i))
			#for smaller_ids_i in input_ids_i:
			#	texts.append(tokenizer.decode(smaller_ids_i))
			

	#print(tokenizer("".join(texts)))

	print("\n\n\n==========================\n\n\n")
	if args.max_token == None:
		print("no max token length limit, and the text is: {}\n".format(texts))
	else:
		print("max token length is {} , and the text is: {}\n".format(args.max_token, texts))



if __name__ == "__main__":
	main(get_args())



