import numpy as np
import gensim


def to_vector(texto: str, model:gensim.models.keyedvectors.KeyedVectors, embeddingLength: int) -> np.ndarray:
    """ 
    Receives a sentence string along with a word embedding model and 
    returns the vector representation of the sentence
    """
    tokens = texto.split() # splits the text by space and returns a list of words
    vec = np.zeros(embeddingLength) # creates an empty vector of 300 dimensions
    for word in tokens: # iterates over the sentence
        if word in model: # checks if the word is both in the word embedding
            vec += model[word] # adds every word embedding to the vector
    return vec / np.linalg.norm(vec) if np.linalg.norm(vec)>0 else vec # divides the vector by their normal