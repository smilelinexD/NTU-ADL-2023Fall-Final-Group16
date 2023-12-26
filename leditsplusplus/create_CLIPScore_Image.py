import os
import pandas as pd
from PIL import Image

for file in os.listdir(os.path.join('data', 'VisA', 'results')):
    type_name = file.split('.')[0]

    anomaly_dir = os.path.join('metrics', 'VisA_base', 'Anomaly', type_name)
    normal_dir = os.path.join('metrics', 'VisA_base', 'Anomaly', type_name, 'label')
    os.makedirs(normal_dir, exist_ok=True)

    df = pd.read_csv(os.path.join('data', 'VisA', 'results', file))
    for d in df.iterrows():
        data = d[1]
        if data['label'] == 'normal':
            continue
        img_anomaly = Image.open(os.path.join('data', 'VisA', data['image']))
        img_anomaly.save(os.path.join(anomaly_dir, os.path.basename(data['image'])))

        try:
            img_normal = Image.open(os.path.join('data', 'VisA', data['image'].replace('Anomaly', 'Normal')))
            img_normal.save(os.path.join(normal_dir, os.path.basename(data['image'])))
        except:
            img_normal = Image.open(os.path.join('data', 'VisA', os.path.dirname(data['image']).replace('Anomaly', 'Normal'), '0' + os.path.basename(data['image'])))
            img_normal.save(os.path.join(normal_dir, os.path.basename(data['image'])))

