import os, json
import pandas as pd

template = 'a photo of {} with {}'

json_data = []
with open('examples/VisA_GPT.json', 'r') as f:
    infos = json.load(f)
for info in infos:
    type_name = info['image_path'].split('/')[2]
    concepts = info['edit_concept_1']
    if info['edit_concept_2'] != '':
        concepts += ',' + info['edit_concept_2']
    if info['edit_concept_3'] != '':
        concepts += ',' + info['edit_concept_3']
    json_data.append({
        'image_path': info['image_path'],
        'tar_prompt': template.format(type_name, concepts)
    })
# for file in os.listdir(os.path.join('data', 'VisA', 'results')):
#     type_name = file.split('.')[0]
#     df = pd.read_csv(os.path.join('data', 'VisA', 'results', file))
#     for d in df.iterrows():
#         data = d[1]
#         if data['label'] == 'normal':
#             continue
#         json_data.append({
#             'image_path': os.path.join('data', 'VisA', data['image']),
#             'tar_prompt': template.format(type_name, data['label'])
#         })

out_name = 'metrics/VisA_GPT_CLIP_score.json'
with open(out_name, 'w') as f:
    json.dump(json_data, f, indent=2)