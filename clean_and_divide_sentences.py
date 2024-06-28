import json
import pandas as pd

def remove_tex_accents(text):
    """
    Remove LaTeX accents from a text.

    Parameters:
    ===========
        text (str): The text from which to remove LaTeX accents.

    Returns:
    ========
        text (str): The text with LaTeX accents removed.
    """
    tex_accents = {
        r'\`': '',
        r"\'": '',
        r'\^': '',
        r'\~': '',
        r'\=': '',
        r'\u': '',
        r'\v': '',
        r'\.': '',
        r'\H': '',
        r'\r': '',
        r'\c': '',
        r'\d': '',
        r'\b': '',
    }

    for accent, replacement in tex_accents.items():
        text = text.replace(accent, replacement)

    return text


def remove_tex_special_characters(text):
    """
    Remove LaTeX special characters from a text.

    Parameters:
    ===========
        text (str): The text from which to remove LaTeX special characters.

    Returns:
    ========
        text (str): The text with LaTeX special characters removed.
    """
    tex_characters = {
        r'\textbackslash': '\\',
        r'\{': '{',
        r'\}': '}',
        r'\_': '_',
        r'\&': '&',
        r'\%': '%',
        r'\#': '#',
        r'\$': '$',
        r'\textasciitilde': '~',
        r'\textasciicircum': '^',
        r'\textasciigrave': '`',
        r'\textasciimacron': '¯',
        r'\textasciibreve': '˘',
        r'\textasciicaron': 'ˇ',
        r'\textasciibreve': '˘',
    }

    for character, replacement in tex_characters.items():
        text = text.replace(character, replacement)

    return text


def clean_sentences(text, substitute_tex=True):
    """
    Clean a text by removing newlines, tabs, and extra spaces.
    In addition, substitute LaTeX special characters with their ASCII equivalents.

    Parameters:
    ===========
        text (str): The text to be cleaned.
        substitute_tex (bool): If True, substitute LaTeX special characters with their ASCII equivalents.

    Returns:
    ========
        cleaned_text (str): The cleaned text.
    """
    cleaned_text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    if substitute_tex:
        cleaned_text = remove_tex_accents(cleaned_text)
        cleaned_text = remove_tex_special_characters(cleaned_text)

    return cleaned_text


def create_sentence_df(json_file, selected_columns, sentence_column='sentence', ignore_columns=None, output_file=None):
    """
    Clean the text in the selected columns of a JSON file and divide it into sentences.

    Parameters:
    ===========
        json_file (str): The path to the JSON file.
        selected_columns (list): A list of strings with the names of the columns to be cleaned and divided into sentences.
        sentence_column (str): The name of the column that will contain the sentences.
        ignore_columns (list): A list of strings with the names of the columns to ignore.
        output_file (str): The name of the output file. If None, the output file will be named "sentence_" + json_file[:-5] + ".parquet".

    Returns:
    ========
        sentence_df (pd.DataFrame): A DataFrame with the sentences and the metadata.
    """
    
    if output_file is None:
        output_file = "sentence_" + json_file[:-5] + ".parquet"
    
    with open(json_file) as file:
        data = json.load(file)
    
    sentence_data = []
    
    # for each entry in the JSON file...
    for entry in data:
        sentences = []
        
        # ...for each column with sentences...
        for column in selected_columns:

            # ...clean the text and divide it into sentences...
            if column in entry:
                text = entry[column]
                cleaned_text = clean_sentences(text, substitute_tex=True)
                entry[column] = cleaned_text

                # ... and add the sentences to the list
                _sentences = [sent.strip() for sent in cleaned_text.split('.')]
                sentences.extend([sent for sent in _sentences if len(sent) > 0])

        # entry['sentences'] = sentences

        # for each sentence in the list...
        for s in sentences:
            # ...create a new entry with the sentence and the metadata...
            sentence_entry = {key: entry[key] for key in entry if key not in selected_columns}
            sentence_entry[sentence_column] = s
            
            # ...and add it to the list
            sentence_data.append(sentence_entry)
    
    # Transform the list of entries into a DataFrame
    sentence_df = pd.DataFrame(sentence_data)

    # Remove ignored columns
    if ignore_columns is not None:
        sentence_df = sentence_df.drop(columns=ignore_columns)

    # Categorize the columns
    keys = sentence_df.keys()
    metadata_columns = [key for key in keys if key != sentence_column]
    sentence_df = sentence_df.astype({column: 'category' for column in metadata_columns})
    sentence_df = sentence_df.astype({sentence_column: 'string'})

    # Save the DataFrame to a parquet file
    sentence_df.to_parquet(output_file, index=False)

    return sentence_df