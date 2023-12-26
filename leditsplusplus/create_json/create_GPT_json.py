import os, json
import pandas as pd

json_data = []
already_in = {}
for file in os.listdir(os.path.join('data', 'VisA', 'results')):
    df = pd.read_csv(os.path.join('data', 'VisA', 'results', file))
    for d in df.iterrows():
        data = d[1]
        if data['label'] == 'normal':
            continue
        concepts = data['label'].split(',')
        for concept in concepts:
            if concept in already_in:
                continue
            json_data.append(concept)
            already_in[concept] = True

with open('GPT.json', 'w') as f:
    json.dump(json_data, f)
