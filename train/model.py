"""
This module construct the model
"""
import tensorflow_hub as hub
import tensorflow as tf


# to speed up training 2x T4 GPUs (offered by Kaggle) is used; mirror strategy is used
strategy = tf.distribute.MirroredStrategy()
print("DEVICES AVAILABLE: {}".format(strategy.num_replicas_in_sync))


def build_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    load_locally = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
    hub_url = "https://tfhub.dev/google/sentence-t5/st5-base/1"
    encoder = hub.KerasLayer(
        hub_url, load_options=load_locally, trainable=False, name="ST5_base_encoder"
    )

    output = encoder(text_input)[0]
    output = tf.keras.layers.Dropout(rate=0, seed=0)(output)
    output = tf.keras.layers.Dense(128, activation=None)(output)
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(
        output
    )  # L2 normalize embeddings
    return tf.keras.Model(text_input, output)
