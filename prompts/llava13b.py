from dotenv import load_dotenv
import replicate
import pandas as pd
import os
import json


def img_caption(imgpath):
    output = replicate.run(
        "yorickvp/llava-13b:e272157381e2a3bf12df3a8edd1f38d1dbd736bbb7437277c8b34175f8fce358",
        input={
            "image": open(imgpath, "rb"),
            "prompt": "Point out the anomaly defect in the picture in a sentence."
        }
    )
    # The yorickvp/llava-13b model can stream output as it's running.
    # The predict method returns an iterator, and you can iterate over that output.
    output_str = ""
    for item in output:
        # https://replicate.com/yorickvp/llava-13b/api#output-schema
        output_str += item
    print(output_str)
    return output_str


def extend_list(labels):
    seperated_labels = []
    for l in labels:
        for s in l.split(","):
            if(s not in seperated_labels):
                seperated_labels.append(s)
    return seperated_labels


def list_subsets(folder):
    subfolders = ["candle", "cashew", "fryum", "macaroni1",
                    "macaroni2", "pcb1", "pcb2", "pcb3",
                    "pcb4", "pipe_fryum", "capsules", "chewinggum"]
    return subfolders


def read_csv_and_save(root, output_dir="llava_syn"):
    subsets = list_subsets(root)

    label_dic = {}
    for subset in subsets:
        print("f{subset}:")
        data = pd.read_csv(f'./labels/{subset}.csv')
        images = data['image'].tolist()

        for img in images:
            imgpath = os.path.join(root, img)
            caption = img_caption(imgpath)
            label_dic[img] = caption

    with open(f"llava_syn/all.json", "w") as outfile: 
        json.dump(label_dic, outfile, indent=4)


if __name__ == '__main__':
    load_dotenv()
    read_csv_and_save("../VisA")