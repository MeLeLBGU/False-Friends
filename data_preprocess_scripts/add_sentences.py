import os
import re
from idlelib.iomenu import encoding

import pandas as pd
# Get the absolute path to the directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path to the 'thesis' directory (move one level up)
thesis_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

ff_data_dir = f"{thesis_dir}/ff_data/"

ff_data_dirs = [f"{ff_data_dir}/de_ff.csv", f"{ff_data_dir}/swedish_ff.csv", f"{ff_data_dir}/spanish_ff.csv",
                f"{ff_data_dir}/italian_ff.csv", f"{ff_data_dir}/french_ff.csv", f"{ff_data_dir}/romanian_ff.csv"]

def add_sentences_to_csv(sentence_path_en,sentence_path_l2, csv_path, l2):
    en_sentences = get_sentences(sentence_path_en)
    l2_sentences = get_sentences(sentence_path_l2)
    df = pd.read_csv(csv_path)
    df = add_sentences_to_df("en sentence", en_sentences, df)
    df = add_sentences_to_df(f"{l2} sentence", l2_sentences, df)
    df.to_csv(csv_path, index=False)
    

def get_sentences(sentence_path):
    with open(sentence_path, 'r', encoding='utf-8-sig') as f:
        sentences = f.readlines()
        data = dict()
        for i in range(len(sentences)):
            if i % 3 == 0:
                cur_word = sentences[i].strip()
                data[cur_word] = []
            else:
                data[cur_word].append(re.sub(r'^\d+\.', '', sentences[i]).strip().lower())
    return data

def add_sentences_to_df(column_name, data, df):
    
    df[f"{column_name}1"] = None
    df[f"{column_name}2"] = None
    for index, row in df.iterrows():
        cur_word = row["False Friend"]
        df.at[index, f"{column_name}1"] = data[cur_word][0]
        df.at[index, f"{column_name}2"] = data[cur_word][1]
    return df
    
add_sentences_to_csv(f"{thesis_dir}/ff_sentences_data/romanian_ff_en.txt", f"{thesis_dir}/ff_sentences_data/romanian_ff.txt",
                     f"{thesis_dir}/ff_data/romanian_ff.csv", "ro")

add_sentences_to_csv(f"{thesis_dir}/ff_sentences_data/spanish_ff_en.txt", f"{thesis_dir}/ff_sentences_data/spanish_ff.txt",
                     f"{thesis_dir}/ff_data/spanish_ff.csv", "es")

add_sentences_to_csv(f"{thesis_dir}/ff_sentences_data/swedish_ff_en.txt", f"{thesis_dir}/ff_sentences_data/swedish_ff.txt",
                     f"{thesis_dir}/ff_data/swedish_ff.csv", "se")

add_sentences_to_csv(f"{thesis_dir}/ff_sentences_data/german_ff_en.txt", f"{thesis_dir}/ff_sentences_data/german_ff.txt",
                     f"{thesis_dir}/ff_data/de_ff.csv", "de")

# add_sentences_to_csv(f"{thesis_dir}/ff_sentences_data/italian_ff_en.txt", f"{thesis_dir}/ff_sentences_data/italian_ff.txt",
#                      f"{thesis_dir}/ff_data/italian_ff.csv", "it")
#
# add_sentences_to_csv(f"{thesis_dir}/ff_sentences_data/french_ff_en.txt", f"{thesis_dir}/ff_sentences_data/french_ff.txt",
#                      f"{thesis_dir}/ff_data/french_ff.csv", "fr")