"""
Second-stage fine-tuning using source-target pair
In this stage further fine-tuning is done on the FC-connected layer with source codes and targets codes
"""

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from keras import backend as K

from model import strategy
from train_utils import plot_history

DATA_DIR = "datasets/"
MODEL_DIR = "saved_models/"

# load data
mimc_df = pd.read_csv("mimc_df.csv")
mimic_loinc_index = mimc_df.loc[:, "LOINC_INDEX"].to_list()

source_target_2nd_stage_df = pd.read_csv(DATA_DIR + "source_target_2nd_stage_df.csv")

# Kfold results
kf = KFold(n_splits=5, random_state=0, shuffle=True)

for fold, (train_index, test_index) in enumerate(kf.split(mimic_loinc_index)):
    with strategy.scope():
        # load model trained from first-stage
        first_stage_model_saved = tf.keras.models.load_model(
            "first_stage_model_epoch_20"
        )

        # add none-zero dropout rate
        first_stage_model_saved.layers[2].rate = 0.2

        # reduced reduced learning rate of 1e-5
        K.set_value(first_stage_model_saved.optimizer.learning_rate, 1e-5)

    # train/val/test split
    mimic_loinc_index_train = [mimic_loinc_index[i] for i in train_index]
    mimic_loinc_index_test = [mimic_loinc_index[i] for i in test_index]
    mimic_loinc_index_train, mimic_loinc_index_val = train_test_split(
        mimic_loinc_index_train, test_size=0.2, random_state=0, shuffle=False
    )

    train_df = source_target_2nd_stage_df[
        source_target_2nd_stage_df["LOINC_INDEX"].apply(
            lambda x: x in mimic_loinc_index_train
        )
    ]
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_df["LOINC_NAME"], train_df["LOINC_INDEX"])
    )
    train_dataset = train_dataset.batch(600)

    val_df = source_target_2nd_stage_df[
        source_target_2nd_stage_df["LOINC_INDEX"].apply(
            lambda x: x in mimic_loinc_index_val
        )
    ]
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_df["LOINC_NAME"], val_df["LOINC_INDEX"])
    )
    val_dataset = val_dataset.batch(600)

    history_2nd_stage = first_stage_model_saved.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
    )

    # save model
    first_stage_model_saved.save(f"{MODEL_DIR}second_stage_model_fold_{fold}")

    # plot learning curve
    plot_history(history_2nd_stage, f"training_plot_2nd_stage_kfold_{fold}.png")
