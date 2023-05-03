import os

"""
    1. Our model is implemented using PyTorch and trained using NVIDIA RTX 3090 GPU.
    2. If you use other environments, you need to adjust the prompt length to get similar results as reported in the paper.
"""
dataset = "CoNLL-2003"
gpu_id = "0"
k = 5
bert_name = "bert"
log_path = f"/main/{dataset}"
if not os.path.exists(f"./logger/{log_path}"):
    os.makedirs(f"./logger/{log_path}")
for comfileName in os.listdir(f"./data/{dataset}"):
    if ".ner" not in comfileName or f"train_{k}_" not in comfileName:
        continue
    promptLen = comfileName.split(".")[0].split("_")[-1] # Modify prompt length here.
    fileName = comfileName.split(".")[0]
    os.system(f"python main.py --datapath ./data --dataset {dataset} --train_text {fileName}.words --train_ner {fileName}.ner --pre_seq_len {promptLen} --gpu_id {gpu_id} --log_path {log_path} --base_model {bert_name} --fewShot {k}")
