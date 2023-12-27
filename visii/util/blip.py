# import os
# os.environ['HF_HOME'] = "cache"
# os.environ['HF_DATASETS_CACHE'] = "cache/data"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "cache/hub"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from PIL import Image
from os.path import join
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def blip_caption(img_path = "../images/pcb1/0_0.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name_or_path = "Salesforce/blip2-flan-t5-xl"
    processor = Blip2Processor.from_pretrained(model_name_or_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name_or_path, \
        load_in_8bit=True, \
        torch_dtype=torch.float16, device_map="auto"
    )

    image = Image.open(img_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


# generated_text = blip_caption()
# print(f"CAPTION : {generated_text}")