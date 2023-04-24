import numpy as np
import tensorflow as tf


def eval(model, source_df, target_df):
    # Embeddings of LOINC code - LONG_COMMON_NAME using official LOINC table
    target_sentences = tf.constant(
        target_df["LONG_COMMON_NAME"].apply(lambda x: x.lower()).to_list()
    )
    target_embeds = model.predict(
        target_sentences,
        batch_size=600,
    )

    # Embeddings of source code based in MIMC data
    source_sentences = tf.constant(source_df["SOURCE_CODE"].to_list())
    source_embeds = model.predict(
        source_sentences,
        batch_size=600,
    )

    # calculating cosine similarity for each potential source-target pair;
    # note that tensor multiplication is used here instead of for loop for performance
    y_pred = tf.matmul(source_embeds[0], target_embeds[0], transpose_b=True)

    # Encode ground truth labels with one-hot
    y_true = np.zeros((len(source_embeds), len(target_embeds)))
    loinc_num_to_index = {
        k: v for k, v in zip(target_df["LOINC_NUM"].to_list(), range(len(source_df)))
    }
    for i in range(len(source_embeds)):
        code_true = source_df.iloc[i]["LOINC_CODE"]
        index = loinc_num_to_index[code_true]
        y_true[i][index] = 1

    # calculate top K accuracy
    for k in (1, 3, 5):
        m = tf.keras.metrics.TopKCategoricalAccuracy(
            k=k, name="top_k_categorical_accuracy", dtype=None
        )
        m.update_state(y_true, y_pred)
        print(f"top {k} accuracy {m.result().numpy()}")
