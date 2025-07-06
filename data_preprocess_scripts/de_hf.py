import csv
import pandas as pd


df = pd.read_parquet("hf://datasets/aari1995/false_friends_en_de/data/train-00000-of-00001-957eef130c71ea88.parquet")
ff_df = {"False Friend": [], "Correct Translation": [], "Wrong Translation": []}
# Filtering out repetitions and words that are not written exactly the same
for index, row in df.iterrows():
    if row["False Friend"].lower() == row["Wrong English Translation"].lower() and row["False Friend"].lower() not in ff_df["False Friend"] and len(row["False Friend"].split(" ")) == 1:
        ff_df["False Friend"].append(row["False Friend"].lower())
        ff_df["Correct Translation"].append(row["Correct English Translation"].lower())
        ff_df["Wrong Translation"].append(row["Wrong English Translation"].lower())
de_ff_df = pd.DataFrame(ff_df)
file_name = f"C:/Users/halor/Desktop/Masters_degree/Melel/thesisV01/ff_data/de_ff.csv"
de_ff_df.to_csv(file_name, encoding="utf-8", index=False)
