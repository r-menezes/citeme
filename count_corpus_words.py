from tqdm import tqdm
import os


def count_words_in_json_fields(json_file, selected_columns):
    """
    Count the number of words in the selected columns of a JSON file.

    Parameters:
    ===========
        json_file (str): The path to the JSON file.
        selected_columns (list): A list of strings with the names of the columns to be cleaned and divided into sentences.

    Returns:
    ========
        word_count (int): The number of words in the selected columns.
        entry_count (int): The number of entries in the JSON file.
        size (str): The size of the JSON file in human-readable format.
    """
    
    with open(json_file) as file:
        data = json.load(file)
    
    word_count = 0
    
    # for each entry in the JSON file...
    for entry in tqdm(data):
        
        # ...for each column with sentences...
        for column in selected_columns:

            # ...and count the number of words in the text
            if column in entry:
                text = entry[column]
                word_count += len(text.split())
    
    return word_count, len(data), f"{os.path.getsize(json_file) / 1024 / 1024:.2f} MB"

if __name__ == "__main__":
    import json
    import sys

    json_file = sys.argv[1]
    selected_columns = sys.argv[2:]

    word_count, entry_count, size = count_words_in_json_fields(json_file, selected_columns)
    print(f"Word count: {word_count}")
    print(f"Entry count: {entry_count}")
    print(f"File size: {size}")