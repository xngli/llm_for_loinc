"""
Data augmentation
Here we define the data augmentation steps following what's described in the paper. Here three augmentation steps are used
- character level random deletion
- Word level random swapping
- Insert random words & acronyms from "RELATEDNAMES2"
"""

import numpy as np
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import random

aug_random_char_delete = nac.RandomCharAug(
    action="delete",
    aug_char_min=1,
    aug_char_max=1,
    aug_word_min=1,
    aug_word_max=1,
)

aug_random_word_swap = naw.RandomWordAug(
    action="swap",
    aug_min=1,
    aug_max=1,
)


def aug_related_word_insert(original, related):
    splitted = original.split(" ")
    words_to_insert = related.split(";")

    # pick random words in words_to_insert
    word_to_insert = random.choice(words_to_insert)

    # pick random index to insert the word
    index_to_insert = random.randint(0, len(splitted))

    splitted.insert(index_to_insert, word_to_insert)
    transformed = " ".join(splitted)
    return transformed


def augment_steps(row, row_name):
    if type(row[row_name]) is not str:
        return np.nan

    if type(row["RELATEDNAMES2"]) is str:
        x = aug_related_word_insert(row[row_name], row["RELATEDNAMES2"])
    else:
        x = row[row_name]

    x = aug_random_char_delete.augment(x)
    x = aug_random_word_swap.augment(x)
    return x[0]
