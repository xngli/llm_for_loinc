import pandas as pd
from sklearn.model_selection import KFold
import tensorflow as tf

from preprocessing.data_utils import augment_steps
from test_utils import test_model

DATA_DIR = "datasets/"
MODEL_DIR = "saved_models/"

# load data
mimc_df = pd.read_csv("mimc_df.csv")
mimic_loinc_index = mimc_df.loc[:, "LOINC_INDEX"].to_list()

loinc_in_mimic_df = pd.read_csv(DATA_DIR + "loinc_in_mimic_df.csv")

# Test with augmentation
# Kfold results
kf = KFold(n_splits=5, random_state=0, shuffle=True)

for fold, (train_index, test_index) in enumerate(kf.split(mimic_loinc_index)):
    # load model trained from 2nd-stage
    second_stage_model_saved = tf.keras.models.load_model(
        f"{MODEL_DIR}second_stage_model_skip_first_stage_fold_{fold}"
    )

    # train/val/test split
    mimic_loinc_index_test = [mimic_loinc_index[i] for i in test_index]

    # test data without augmentation
    test_df = mimc_df.iloc[test_index]

    test_df_with_related_names = test_df.merge(
        mimc_df[["LOINC_CODE", "RELATEDNAMES2"]],
        on="LOINC_CODE",
        how="left",
    )
    test_df_augmented = []

    # augment test set by 100x
    for i in range(100):
        df_to_add = test_df_with_related_names.apply(
            lambda x: augment_steps(x, "SOURCE_CODE"), axis=1
        ).to_frame("SOURCE_CODE")
        df_to_add["LOINC_INDEX"] = test_df_with_related_names["LOINC_INDEX"]
        df_to_add["LOINC_CODE"] = test_df_with_related_names["LOINC_CODE"]
        test_df_augmented.append(df_to_add)

    test_df_augmented = pd.concat(test_df_augmented, axis=0).drop_duplicates()

    # get accuracy
    test_model(second_stage_model_saved, test_df_augmented, loinc_in_mimic_df)
