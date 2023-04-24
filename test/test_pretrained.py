"""
Performance using pre-trained LLM
In this step the pre-trained T5 model is loaded and used to extract embeddings from the source & target codes. 
The embedding is later used for calculating cosine similarity which is used for top-k accuracy calculation.
"""

import pandas as pd
import tensorflow_hub as hub

from test_utils import test_model

DATA_DIR = "datasets/"

# load model
hub_url = "https://tfhub.dev/google/sentence-t5/st5-base/1"
encoder = hub.KerasLayer(hub_url)

# load data
mimc_df = pd.read_csv(DATA_DIR + "mimc_df.csv")
loinc_in_mimic_df = pd.read_csv(DATA_DIR + "loinc_in_mimic_df.csv")

# get accuracy
test_model(encoder, mimc_df, loinc_in_mimic_df)
