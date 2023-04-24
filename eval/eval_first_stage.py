import pandas as pd
import tensorflow as tf

DATA_DIR = "datasets/"
MODEL_DIR = "saved_models/"

# load model
first_stage_model_loaded = tf.keras.models.load_model(
    MODEL_DIR + "first_stage_model_epoch_20"
)

# load data
mimc_df = pd.read_csv(DATA_DIR + "mimc_df.csv")
loinc_in_mimic_df = pd.read_csv(DATA_DIR + "loinc_in_mimic_df.csv")

# get accuracy
eval(first_stage_model_loaded, mimc_df, loinc_in_mimic_df)
