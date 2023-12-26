# Text-based Image Editing

## Project Structure
* leditsplusplus/ is copied from official github, modified for generating our anomalous images
```
+ inference.py
+ create_json/*.py
+ metrics/CLIP_I.py
+ metrics/CLIP_score.py
```
* change_to_text.py, clip_embed.py are used for reverting prompt embeddings to text

## How to run
* Create JSON file for Ledits++ inference, files are task-specific
```bash=
python leditsplusplus/create_json/*.py
```
* Inference Ledits++
```bash=
python leditsplusplus/inference.py --input_prompt_path <path/to/json>
```
* Evaluation (CLIP-I)
```bash=
python leditsplusplus/metrics/CLIP_I.py --base_image_dir <path/to/dir>
```
* Evaluation (CLIP-T)
```bash=
python leditsplusplus/metrics/CLIP_score.py --prompt <path/to/json>
```
---
original README.md from leditsplusplus  
Paper: arxiv.org/abs/2311.16711