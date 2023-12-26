import argparse
import glob
import json
import os

import torch
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import (dot_score, normalize_embeddings,
                                        semantic_search)
from visii import StableDiffusionVisii


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--checkpoint_number', type=str, default='best')
    parser.add_argument('--log_dir', type=str, default='./logs/ip2p_painting1_0_0.png')
    parser.add_argument('--out_dir', type=str, default='./prompt_text/')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--max_token', type=int, default = 15)
    return parser.parse_args()






def main(args):
    pipe = StableDiffusionVisii.from_pretrained(args.model_id, torch_dtype=torch.float32)
    #print(pipe.text_encoder)
    model = pipe.text_encoder.to(args.device)
    tokenizer = pipe.tokenizer

    checkpoint = os.path.join(args.log_dir, 'prompt_embeds_{}.pt'.format(args.checkpoint_number))
    opt_embs = torch.load(checkpoint, map_location=args.device)
    #print(opt_embs)
    #print(model.text_model.embeddings.token_embedding.weight)
    #print(opt_embs.shape, model.text_model.embeddings.token_embedding.weight.shape)
    #print(opt_embs[0])
    #for i in range(77):
    #    print(opt_embs[0][i][:10])
    
    curr_embeds = opt_embs[0] ## [1] [2] is negative prompt
    if args.max_token != None:
        curr_embeds = curr_embeds[:args.max_token]
    
    #print(curr_embeds.shape)
    curr_embeds = normalize_embeddings(curr_embeds)

    embedding_matrix = model.text_model.embeddings.token_embedding.weight

    with torch.no_grad():
        for i in range( embedding_matrix.shape[0]):
            #print( tokenizer(tokenizer.decode(torch.tensor([i+1000]))) )
            #time.sleep(1)
            if ( tokenizer(tokenizer.decode(torch.tensor([i])))['input_ids'][1] != i ):
                embedding_matrix[i]=torch.zeros_like(embedding_matrix[i])
    
    embedding_matrix = embedding_matrix.detach()


    embedding_matrix = normalize_embeddings(embedding_matrix)



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
            #    texts.append(tokenizer.decode(smaller_ids_i))
            

    #print(tokenizer("".join(texts)))

    print("\n\n\n==========================\n\n\n")
    if args.max_token == None:
        print("no max token length limit, and the text is: {}\n".format(texts))
    else:
        print("max token length is {} , and the text is: {}\n".format(args.max_token, texts))




if __name__=="__main__":
    main(get_args())