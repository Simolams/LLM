import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckduckgo_search
from duckduckgo_search import DDGS
import regex as re
from datasets import load_dataset
import transformers
from transformers import pipeline
from difflib import SequenceMatcher



def search_google(query):

    #content = ""
    content = []
    results = DDGS().text(query, max_results=10)

    for text in results : 
        #content+="\n"+text['body']
        sub_text = re.sub(r'([^\s\w]|_)+', '', text['body'])
        content.append(sub_text)

    return content




def results_parser(output):

    formated_output = []

    for dic in output : 
        formated_output.append(dic['answer'].lstrip())
    
    return formated_output





def QA_answer(query,qa_model,best_score=False):

    ## Parallelization
    contexts = search_google(query)
    # Create a batch of inputs
    batch = [{"question": query, "context": context} for context in contexts]

    # Use the model to process the batch
    results = qa_model(batch)
    output = results_parser(results)
    if best_score : 
        return pd.DataFrame(results).sort_values(by='score',ascending=False).answer.iloc[0].lstrip()

    return output


def answers_index(query,answer):

    """cet fonction permet de savoir si la réponse existe déja dans le contexte ou non 
    True si answer in context false sinon
    ça permet de mieux evaluer les modèles 
    """

    contexts = search_google(query)
    # Create a batch of inputs
    
    for text in contexts : 
        if answer in text : 
            return True
    
    return False
    






clean  = lambda text : [txt.replace(',','').replace('(','').replace(')','').lstrip() for txt in text]



############## For model evaluation ###############################
def calculate_similarity(word1, word2):
    """
    Calculate the similarity rate between two words based on SequenceMatcher.
    
    Args:
        word1 (str): The first word.
        word2 (str): The second word.
    
    Returns:
        float: Similarity rate between 0.0 (no similarity) and 1.0 (exact match).
    """
    return SequenceMatcher(None, word1, word2).ratio()
