"""
This module contains code that preprocess the data.
"""
import pandas as pd
from tqdm.notebook import tqdm

from preprocessing.data_utils import augment_steps

DATA_DIR = "datasets/"

########################### Load MIMIC-III data ##############################
# This loads the `D_LABITEMS` tables from MIMC data that's used in the paper
mimc_df = pd.read_csv("/kaggle/input/llm-for-loinc/D_LABITEMS.csv")

# generates source code by concatenating the "label" & "fluid" colummns per the paper
mimc_df["SOURCE_CODE"] = mimc_df.apply(
    lambda x: (f"{x['LABEL']} {x['FLUID']}").lower(), axis=1
)

# drop duplicates
mimc_df = (
    mimc_df[["SOURCE_CODE", "LOINC_CODE"]]
    .dropna()
    .drop_duplicates()
    .drop_duplicates("SOURCE_CODE", keep=False)
)


### Load official LOINC table (target codes)
# This loads the target LOINC codes from the offical table

loinc_df = pd.read_csv("/kaggle/input/llm-for-loinc/Loinc.csv")

##################  Prepare data to use pre-trained model ######################
# To test the pre-trained model, we need the MIMIC dataset, as well as a subset of the target code table
# whose LOINC code match that of in the MIMC dataset. No data augmentation is needed here.

# loinc code in the mimic-iii table
mimic_loinc_list = mimc_df["LOINC_CODE"].to_list()
loinc_in_mimic_df = loinc_df[
    loinc_df["LOINC_NUM"].apply(lambda x: x in mimic_loinc_list)
][["LOINC_NUM", "LONG_COMMON_NAME", "DisplayName", "SHORTNAME", "RELATEDNAMES2"]]

loinc_in_mimic_df.to_csv(DATA_DIR + "loinc_in_mimic_df.csv")

##################### Prepare data for stage one fine-tuning ####################

# Following the paper to extract a subset of LOINC codes in the laboratory and clinical categories
# From LOINC official user guide 1=Laboratory class; 2=Clinical class; 3=Claims attachments; 4=Surveys
loinc_lab_clinical_df = loinc_df[loinc_df["CLASSTYPE"].apply(lambda x: x <= 2)][
    ["LOINC_NUM", "LONG_COMMON_NAME", "DisplayName", "SHORTNAME", "RELATEDNAMES2"]
].rename(columns={"LOINC_NUM": "LOINC_CODE"})

# encode target as integer (this is needed to use Tripletloss in Tensorflow)
loinc_num_to_index = {
    k: v
    for k, v in zip(
        loinc_lab_clinical_df["LOINC_CODE"].unique(),
        range(loinc_lab_clinical_df["LOINC_CODE"].nunique()),
    )
}
loinc_lab_clinical_df.loc[:, "LOINC_INDEX"] = loinc_lab_clinical_df["LOINC_CODE"].apply(
    lambda x: loinc_num_to_index[x]
)

# apply the data augmentation step (this only needs to be run once as the results will be saved as csv)
for col in tqdm(["LONG_COMMON_NAME", "DisplayName", "SHORTNAME"]):
    loinc_lab_clinical_df[f"{col}_AUGMENTED"] = loinc_lab_clinical_df.apply(
        lambda x: augment_steps(x, col), axis=1
    )

# melt the dataframe to long format
loinc_lab_clinical_df_long = loinc_lab_clinical_df.melt(
    id_vars="LOINC_INDEX",
    value_vars=[
        "LONG_COMMON_NAME_AUGMENTED",
        "DisplayName_AUGMENTED",
        "SHORTNAME_AUGMENTED",
    ],
    value_name="LOINC_NAME",
)
loinc_lab_clinical_df_long = loinc_lab_clinical_df_long[
    loinc_lab_clinical_df_long["LOINC_NAME"].notna()
].sort_values("LOINC_INDEX")
loinc_lab_clinical_df_long.loc[:, "LOINC_NAME"] = loinc_lab_clinical_df_long[
    "LOINC_NAME"
].apply(lambda x: x.lower())


loinc_lab_clinical_df_long.to_csv(DATA_DIR + "Loinc_lab_clinical_augmented.csv")

######################## Data preparation for 2nd stage fine-tuning ########################
# encode the target as integer (using same encoding map) for MIMC dataset
mimc_df.loc[:, "LOINC_INDEX"] = mimc_df["LOINC_CODE"].apply(
    lambda x: loinc_num_to_index[x]
)

# Next the "RELATEDNAMES2" column is added to the mimc_df
mimc_df = mimc_df.merge(
    loinc_lab_clinical_df[["LOINC_INDEX", "RELATEDNAMES2"]],
    on="LOINC_INDEX",
    how="left",
)

# augment 3 times
for i in tqdm(range(1, 1 + 3)):
    mimc_df[f"SOUCE_CODE_AUGMENTED_{i}"] = mimc_df.apply(
        lambda x: augment_steps(x, "SOURCE_CODE"), axis=1
    )

# save file
mimc_df.to_csv(DATA_DIR + "mimc_df.csv")

# melt to long format
mimc_df_with_related_names_long = mimc_df.melt(
    id_vars="LOINC_INDEX",
    value_vars=[
        "SOUCE_CODE_AUGMENTED_1",
        "SOUCE_CODE_AUGMENTED_2",
        "SOUCE_CODE_AUGMENTED_3",
    ],
    value_name="LOINC_NAME",
)
mimc_df_with_related_names_long = mimc_df_with_related_names_long[
    mimc_df_with_related_names_long["LOINC_NAME"].notna()
].sort_values("LOINC_INDEX")
mimc_df_with_related_names_long.loc[:, "LOINC_NAME"] = mimc_df_with_related_names_long[
    "LOINC_NAME"
].apply(lambda x: x.lower())

# For 2nd stage fine-tuning we only need the part of loinc_lab_clinical_df_long whose LOINC code appears in MIMC
mimic_loinc_index = mimc_df.loc[:, "LOINC_INDEX"].to_list()
loinc_lab_clinical_df_long_2nd_stage = loinc_lab_clinical_df_long[
    loinc_lab_clinical_df_long["LOINC_INDEX"].apply(lambda x: x in mimic_loinc_index)
]
loinc_lab_clinical_df_long_2nd_stage = loinc_lab_clinical_df_long_2nd_stage.astype(
    {"LOINC_INDEX": "int64"}
)
loinc_lab_clinical_df_long_2nd_stage

# Combine source and target dataset (post augmentation) to make the trainning/eval data for 2nd stage fine-tuning
source_target_2nd_stage_df = pd.concat(
    [loinc_lab_clinical_df_long_2nd_stage, mimc_df_with_related_names_long], axis=0
).sort_values("LOINC_INDEX")

source_target_2nd_stage_df.to_csv(DATA_DIR + "source_target_2nd_stage_df.csv")
