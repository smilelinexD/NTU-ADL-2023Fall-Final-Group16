import os
os.environ['HF_HOME'] = "cache"
os.environ['HF_DATASETS_CACHE'] = "cache/data"
os.environ["HUGGINGFACE_HUB_CACHE"] = "cache/hub"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
from sentence_transformers import util, SentenceTransformer
import numpy as np


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword)


def cal_w2w(gt_label, syn_labels):
    cos_sim = cosine_sim()
    vectorizer = text2vector()

    # Get the similarity scores
    gt = vectorizer(gt_label)
    sim = [cos_sim(vectorizer(sl), gt).item() for sl in syn_labels]
    sim = np.array(sim)
    # top_indices = np.argpartition(sim, 3)[:3]
    return np.array(syn_labels)[np.argmin(sim)]


def paraphrase(subset, label):
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws").to('cuda')
    
    text =  f"paraphrase: a photo of {subset} with {label} </s>"
    encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")

    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_masks,
        max_length=256,
        do_sample=True,
        top_k=120,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=5
    )

    res = []
    for output in outputs:
        line = tokenizer.decode(output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
        res.append(line)
    res3 = cal_w2w(text, res)
    return res3


def list_subsets(folder):
    subfolders = ["candle", "cashew", "fryum", "macaroni1",
                    "macaroni2", "pcb1", "pcb2", "pcb3",
                    "pcb4", "pipe_fryum", "capsules", "chewinggum"]
    return subfolders


def read_csv_and_save(subsets, output_dir="t5_syn"):
    label_dic = {}

    for subset in subsets:
        print("f{subset}:")
        data = pd.read_csv(f'./labels/{subset}.csv')
        labels = data['label'].tolist()
        images = data['image'].tolist()

        for label, image in zip(labels, images):
            if(label == 'normal' or label == 'other'): 
                continue
            # syn_labels = [p for p in paraphrase(label)]
            # label_dic[label] = syn_labels
            syn_sentenc = paraphrase(subset, label)
            label_dic[image] = syn_sentenc
        print(label_dic)

    with open(f"t5_syn/all_sentenc.json", "w") as outfile: 
        json.dump(label_dic, outfile, indent=4)


def extend_list(labels):
    seperated_labels = []
    for l in labels:
        for s in l.split(","):
            if(s not in seperated_labels):
                seperated_labels.append(s)
    return seperated_labels


if __name__ == '__main__':
    subsets = list_subsets("../VisA")
    read_csv_and_save(subsets)

    
