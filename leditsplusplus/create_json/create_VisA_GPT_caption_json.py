import os, json
import pandas as pd

out_name = 'VisA_T5_caption.json'
img_out_dirname = 'Normal'

add0 = ['candle', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4']
processed_type = ['candle', 'capsules', 'cashew', 'chewinggum']

with open('all_sentence.json', 'r') as f:
    captions = json.load(f)

json_data = []
for file in os.listdir(os.path.join('data', 'VisA', 'results')):
    file_type = file.split('.')[0]
    if file_type not in processed_type:
        continue
    df = pd.read_csv(os.path.join('data', 'VisA', 'results', file))
    for d in df.iterrows():
        data = d[1]
        if data['label'] == 'normal':
            continue
        file_type = file.split('.')[0]
        file_path = data['image'].split('/')
        img_name = file_path[-1]
        file_path[3] = 'Normal'
        out_file_name = '0' + img_name if file_type in add0 else img_name
        file_path[4] = out_file_name
        # out_file_name = file_path[-1]
        file_path = os.path.join(*file_path)
        guidance_scale = 15
        json_data.append({
            'image_path': os.path.join('data', 'VisA', file_path),
            'tar_prompt': captions[data['image']],
            'tar_cfg_scale': 15,
            'output_image_path': os.path.join('outputs', 'VisA_T5_caption', img_out_dirname, file_type, out_file_name)
        })

with open(os.path.join('examples', out_name), 'w') as f:
    json.dump(json_data, f, indent=2)
