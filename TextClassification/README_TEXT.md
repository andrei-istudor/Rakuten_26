---
noteId: "a29f4270fd0711f0bc74e5a9575ee52b"
tags: []

---

# Text pipeline

## Exploration (rakuten_exploring)

- reading data
- "smaller" manipulations
- reading language detection from separate df (evaluated in separate script, unfortunately lost during server crash, but simple to rewrite, if needed)
- saving data frame rakuten_explored.csv

## Preprocessing (rakuten_preprocessing)

- read rakuten_explored.csv
- remove HTML noise
- add column with lowercased and stripped text data
- remove duplicates
- quick analysis of processed text
- saving data frame to rakuten_processed.csv

## Translation (rakuten_translation)

- script for translating all non-english text into english
    - translation for full text column (stripped and combined designation and description)
    - translation of each text column individually (for streamlit app)
    - scripts works such that it just runs through without doing anything if saved dataframe of translated texts are already fully translated 

## Splitting (rakuten_split)

- manages the split of the data for both the image and text modelling

## rakuten_modelling_LinearSVM_optim

- script to run a TF-IDF vectorization and LinearSVM classification
    - either on a single vectorizer (word or char)
    - combined vectorizer (char+word)
    - saves runs and metrics

## rakuten_eval_LinearSVM

- evaluation of LinSVM experiments

## BERT

- rakuten_modelling_BERT_optim (modelling using HF platform)
- rakuten_eval_BERT (evaluation of the modelling experiments)
- rakuten_eval_comp (comparison of the evaluations)

## setup_env.ssh

- setup of the linux server to run the BERT scripts

## requirements.txt

- python venv for all scripts except the three BERT scripts

