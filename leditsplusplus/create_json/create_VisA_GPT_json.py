import os, json
import pandas as pd

out_name = 'VisA_GPT1.json'
img_out_dirname = 'Normal'

add0 = ['candle', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4']

with open('GPT.json', 'r') as f:
    synonyms = json.load(f)
# print(synonyms.keys())
synonyms_id = 1

json_data = []
for file in os.listdir(os.path.join('data', 'VisA', 'results')):
    if 'pcb' in file:
        continue
    df = pd.read_csv(os.path.join('data', 'VisA', 'results', file))
    for d in df.iterrows():
        data = d[1]
        if data['label'] == 'normal':
            continue
        type_name = file.split('.')[0]
        concepts = data['label'].split(',')
        
        file_path = data['image'].split('/')
        img_name = file_path[-1]
        file_path[3] = 'Normal'
        out_file_name = '0' + img_name if type_name in add0 else img_name
        file_path[4] = out_file_name
        # out_file_name = file_path[-1]
        file_path = os.path.join(*file_path)
        guidance_scale = 15
        try:
            json_data.append({
                'image_path': os.path.join('data', 'VisA', file_path),
                'edit_concept_1': synonyms[concepts[0].strip()][synonyms_id],
                'edit_concept_2': '' if len(concepts) < 2 else synonyms[concepts[1].strip()][synonyms_id],
                'edit_concept_3': '' if len(concepts) < 3 else synonyms[concepts[2].strip()][synonyms_id],
                'guidnace_scale_1': guidance_scale,
                'guidnace_scale_2': 7 if len(concepts) < 2 else guidance_scale,
                'guidnace_scale_3': 7 if len(concepts) < 3 else guidance_scale,
                'output_image_path': os.path.join('outputs', 'VisA_GPT1', img_out_dirname, type_name, out_file_name)
            })
        except:
            pass

with open(os.path.join('examples', out_name), 'w') as f:
    json.dump(json_data, f, indent=2)
