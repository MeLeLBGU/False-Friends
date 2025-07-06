import os

languages_set = ["English", "French", "Spanish", "German", "Swedish", "Italian", "Romanian"]
language_dict = dict()
for l in languages_set:
    with open(f"../all_words_in_all_languages/{l}/{l}.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()[0].strip().lower().split(",")
    language_dict[l] = set(lines)

for l in languages_set[1:]:
    print(f"FF words English and {l}")
    ff_words = language_dict["English"] & language_dict[l]
    print(ff_words)
    print(f"number of FF words: {len(ff_words)}")