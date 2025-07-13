import os

languages_map = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "se": "Swedish", "it": "Italian", "ro": "Romanian"}

def get_same_words_across_languages(l1, l2):
    with open(f"./all_words_in_all_languages/{languages_map[l1]}/{languages_map[l1]}.txt", 'r', encoding='utf-8') as f1:
        line1 = f1.readlines()[0].strip().lower().split(",")
    l1_words_set = set(line1)
    with open(f"./all_words_in_all_languages/{languages_map[l2]}/{languages_map[l2]}.txt", 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()[0].strip().lower().split(",")
    l2_words_set = set(lines2)
    return l1_words_set & l2_words_set