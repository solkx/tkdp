# [TKDP: A Threefold Knowledge-Enhanced Deep Prompt Tuning for Few-shot Named Entity Recognition](https://ieeexplore.ieee.org/document/10502152)

## Environments

```
- python (3.8)
- cuda (11.0)
- torch (1.7.1)
- pip install -r requirements.txt
```

## Preparation
   * Get dataset 
   * Process them to fit the same format as the example in `data/`
   * Put the processed data into the directory `data/`
   * The file `data/nature_form_label.json` stores the natural language entity types of all data sets
   * Run `sememe_id.py` or `sememe_id_tree.py` obtains the sememe knowledge file

## Training  
```text
python main.py
```
Modify the parameter **dataset** and **k** in the file `main. py` to train different settings or data sets
