import requests
from bs4 import BeautifulSoup
import pandas as pd
import re


def get_ff_languages(row_data, language_set):
    """
    This function receives the data of the 0th column, and extracts the false friend word for each language. The data
    is saved in a dictionary with key=language and value=ff
    :param row_data: the first column data
    :param language_set: languages of interest
    :return: dictionary with key=language and value=ff
    """
    # Regex of 2 groups. 1st group matches the false friends words. 2nd group matches the languages which are in parentheses
    matches = re.findall(r"([^(]+)\s*\(([^)]+)\)", row_data)
    words_by_language = {}
    
    for words, languages in matches:
        # Some ff words can be written in different ways, which are split by the characters ",/"
        ff_words = [w.strip() for w in words.strip().replace("/", ",").split(",") if len(w.strip()) > 0]
        # Some languages are seperated by "," or use the word "and"
        ff_languages = [l.strip() for l in languages.replace("and", ",").split(",")]
        
        for l in ff_languages:
            if l.lower() not in language_set or len(ff_words) == 0:
                continue
            words_by_language[l.lower()] = ff_words
    
    return words_by_language


def ff_filter(lang_ff_dic, row_data):
    """
    This function filters out ff words that are not written exactly the same as English
    :param lang_ff_dic: dictionary with key=language and value=ff
    :param row_data: 1st column data
    :return: updated lang_ff_dic
    """
    keys_to_remove = []
    for key, values in lang_ff_dic.items():
        for v in values:
            # finds if the ff word is in the 1st column data
            if re.search(r"\b" + re.escape(v.lower()) + r"\b", row_data.lower()) is not None:
                lang_ff_dic[key] = v.lower()
                break
            elif v == values[-1]:
                keys_to_remove.append(key)
    for key in keys_to_remove:
        lang_ff_dic.pop(key)
    return lang_ff_dic


def process_row(row_data, language_set):
    """
    This function processes the rows of the ff table by calling different row processing functions, and finally also
    updates the row to its final form: ["False Friend", "Wrong English Translation", "Correct English Translation"]
    :param row_data: the row data from the wikitionary table
    :param language_set: languages of interest
    :return: a dictionary with key=language and value=["False Friend", "Wrong English Translation", "Correct English Translation"]
    """
    lang_ff_dic = get_ff_languages(row_data[0], language_set)
    lang_ff_dic = ff_filter(lang_ff_dic, row_data[1])
    for key, v in lang_ff_dic.items():
        lang_ff_dic[key] = [v, v, row_data[2]]
    return lang_ff_dic


def update_ff_tables(lang_ff_dict, ff_tables, table_cols):
    """
    This function creates and updates a pandas table for each language
    :param lang_ff_dict: a dictionary with key=language and value=["False Friend", "Wrong English Translation", "Correct English Translation"]
    :param ff_tables: a dictionary with key=language and value=pd.DataFrame
    :param table_cols: table columns
    :return: updated ff_tables
    """
    for l, row in lang_ff_dict.items():
        if l not in ff_tables.keys():
            ff_tables[l] = pd.DataFrame(columns=table_cols)
        ff_tables[l].loc[len(ff_tables[l].index)] = row
    return ff_tables


url = "https://en.wiktionary.org/wiki/Appendix:Glossary_of_false_friends"
page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')
# Gets all the tables in the wiktionary web page
wiki_tables = soup.find_all('table', class_='wikitable')
table_cols = ["False Friend", "Wrong Translation", "Correct Translation"]
language_set = {"french", "italian", "latin", "german", "dutch", "hungarian", "finnish", "estonian", "croatian",
                "swedish", "danish",
                "spanish", "portuguese", "esperanto", "polish", "romanian", "indonesian"}

# Creating pandas table for each language
ff_tables = dict()
for t in wiki_tables:
    # Get all the table rows
    table_rows = t.find_all('tr')
    for row in table_rows:
        # Get all the data from the table row, i.e. row[i], 0<i<n
        row_data = row.find_all('td')
        row_data = [data.text.strip() for data in row_data][:3]
        # some rows are empty
        if len(row_data) > 0:
            lang_ff_dict = process_row(row_data, language_set)
            ff_tables = update_ff_tables(lang_ff_dict, ff_tables, table_cols)

for l, ff_df in ff_tables.items():
    file_name = f"C:/Users/halor/Desktop/Masters_degree/Melel/thesisV02/ff_data/{l}_ff.csv"
    ff_df.to_csv(file_name, encoding="utf-8", index=False)