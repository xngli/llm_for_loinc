"""
First-stage fine-tuning using target codes only
In this stage the fine-tuning is done on the FC-connected layer (T5 backbone weights is not changed). 
Note that the fine-tune step is using augmented target codes only from the offical LOINC table.
"""

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

from model import build_model, strategy
from train_utils import plot_history


DATA_DIR = "datasets/"
MODEL_DIR = "saved_models/"
EPOCHS = 20


# construct the model
with strategy.scope():
    first_stage_model = build_model()

    # use adam optimizer and learning rate of 1e-4 as specified in the paper
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
    )

    # here we focus on TripletSemiHardLoss only (original paper studied both semihard &
    # hard triplet search strategies)
    loss = tfa.losses.TripletSemiHardLoss(margin=0.8)

    first_stage_model.compile(
        optimizer=optimizer,
        loss=loss,
    )

# load augmented data
loinc_lab_clinical_df_long = pd.read_csv(DATA_DIR + "Loinc_lab_clinical_augmented.csv")

### train/validation split
train_df, val_df = train_test_split(
    loinc_lab_clinical_df_long, test_size=0.2, random_state=0, shuffle=False
)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df["LOINC_NAME"], train_df["LOINC_INDEX"])
)
train_dataset = train_dataset.batch(600)

val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_df["LOINC_NAME"], val_df["LOINC_INDEX"])
)
val_dataset = val_dataset.batch(600)

### train model
history = first_stage_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
)

# save model
first_stage_model.save(MODEL_DIR + "first_stage_model_epoch_20")

# plot training curve
plot_history(history, "training_plot.png")
