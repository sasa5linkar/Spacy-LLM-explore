import xml.etree.ElementTree as ET
import pandas as pd
import os
import requests
from constants import RESUORCES_DIR, COPRUS_DIR, SPACY_PIPELINE_DIR, api_url
from spacy.tokens import DocBin
from spacy.vocab import Vocab
import numpy as np

def load_data_from_xml(path):
    """
    This function loads data from an XML file into a pandas DataFrame.

    Parameters:
    path (str): The path of the XML file to load.

    Returns:
    DataFrame: A pandas DataFrame containing the data from the XML file.
    """
    # Parse the XML file
    tree = ET.parse(path)
    root = tree.getroot()

    # Create empty lists to store the data
    titles = []
    texts = []
    authors = []
    dates = []
    newspapers = []
    categories = []

    # Loop through the XML elements and extract the data
    for doc in root.findall('document'):
        title = doc.find('Naslov').text
        text = doc.find('Tekst').text
        author = doc.find('Autor').text
        date = doc.find('Datum').text
        newspaper = doc.find('Novina').text
        category = doc.find('RubrikaAA').text

        # Append the data to the lists
        titles.append(title)
        texts.append(text)
        authors.append(author)
        dates.append(date)
        newspapers.append(newspaper)
        categories.append(category)

    # Create a dictionary with the data
    data = {
        'title': titles,
        'text': texts,
        'author': authors,
        'date': dates,
        'newspaper': newspapers,
        'category': categories
    }

    # Create a Pandas DataFrame from the dictionary
    df = pd.DataFrame(data)

    return df

def xml_path (name):
    return os.path.join(RESUORCES_DIR, name + '.xml')
def spacy_folder_path (name):
    spacy_dir= os.path.join(COPRUS_DIR, name)
    if not os.path.exists(spacy_dir):
        os.makedirs(spacy_dir)
    return spacy_dir

def spacy_path (name, filecounter=0):
    return os.path.join(spacy_folder_path(name), name + f'_{filecounter}.spacy')

def spacy_pipeline_path (name):
    return os.path.join(SPACY_PIPELINE_DIR, name)

def save_data(data, path):
    """
    This function saves the data to a .spacy file at the provided path.
    """
    DocBin(docs=data).to_disk(path)

def split_docbin(large_docbin, name, limit=1000):
    """
    This function splits a large DocBin into smaller DocBins and saves them to disk.

    Parameters:
    large_docbin (DocBin): The large DocBin to split.
    name (str): The name of the corpus.
    nlp (Language): The spaCy Language object.
    limit (int): The maximum number of documents to include in each DocBin.
    """

    new_docbin = DocBin()
    counter = 0
    file_counter = 0
    vocab = Vocab()
    for doc in large_docbin.get_docs(vocab):
        new_docbin.add(doc)
        counter += 1

        if counter >= limit:  # Use the default limit if none is provided
            new_docbin.to_disk(spacy_path(name, file_counter))
            counter = 0
            file_counter += 1
            new_docbin = DocBin()

    # Don't forget to save the last DocBin if it's not empty
    if counter > 0:
        new_docbin.to_disk(spacy_path(name, file_counter))

def load_all_docs_from_directory(directory_path):
    """
    This function loads all the documents from a directory of .spacy files into a single DocBin.

    Parameters:
    directory_path (str): The path of the directory containing the .spacy files.

    Returns:
    DocBin: A DocBin containing all the documents from the directory.
    """

    combined_doc_bin = DocBin()  # This will hold all the documents

    # Iterate over files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.spacy'):  # Check if the file is a .spacy file
            file_path = os.path.join(directory_path, filename)
            doc_bin = DocBin().from_disk(file_path)  # Load the DocBin file

            # Merge the loaded DocBin into the combined DocBin
            combined_doc_bin.merge(doc_bin)

    return combined_doc_bin
def get_embeddings(texts):
    response = requests.post(api_url, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()[0]

def get_sentence_embedding(word_embeddings):
    return np.mean(word_embeddings, axis=0)