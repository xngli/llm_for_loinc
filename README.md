# llm_for_loinc
Use LLM for predict standard LOINC code from lab source code

This repo is hosting code for the final project of CSE 6250 at Georgia Tech. The goal is to reproduce key results from the paper [Automated LOINC Standardization Using Pre-trained Large Language Models](https://proceedings.mlr.press/v193/tu22a.html). The code is based on TensorFlow.

## Dependencies
The dependencies are specified in  `requirements.txt`. Run the following to install dependencies

`pip install -r requirements.txt`

## Datasets and preprocessing
There are two datasets required for this project: 1) the offical [LOINC](https://loinc.org/file-access/?download-id=476131) table, and 2) the [MIMIC-III](https://physionet.org/content/mimiciii/1.4/D_LABITEMS.csv.gz) Clinical Database 1.4. Proper registration and/or training is needed to obtain both datasets therefore they are not included in this repository. Once these data are downloaded they should be placed in the `/datasets` folder.

Before model training and testing, run the following to preprocess and augment the data

`python preprocessing/data_processing.py`

## Model training and testing

The model development consists of different stages. Use the following for training and testing for each stage

- Test pre-trained Sentence-T5 model (the module includes code that downloads the pre-trained weights from TensorFlow Hub)

`python test/test_pretained.py`

- Train and test in first stage fine-tuning

```
python train/first_stage.py
python test/test_first_stage.py
```

- Train and test in second stage fine-tuning

`python train/second_stage.py`

`python test/test_second_stage.py`

- Skip first stage fine-tuning and go directly to second stage fine-tuning

`python train/skip_first_stage.py`

`python test/test_skip_stage.py`