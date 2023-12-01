from nltk.stem.snowball import SnowballStemmer
import spacy


class TextCleaner():
    def __init__(self, nlp:str, language:str):
        self.nlp = spacy.load(nlp)
        self.stemmer = SnowballStemmer(language=language)
    
    def clean_text(self, text: str) -> list:
        """
        Recive a string and return the string in tokens without punctuations
        and in lowercase
        """
        # for each token in the sentence add to the list if is not a punctuation
        return [t for t in self.nlp(text.lower()) if not t.is_punct]
        
    def normalize(self, text: str) -> str:
        """
        Recive al list of string and return in one string without stop words
        """
        tokens = self.clean_text(text)
        # for each token if is not a stop word add the word to the list
        words = [t.orth_ for t in tokens]

        # return the tokens in one string
        return(" ".join(words))

    def lemmatize(self, text: str) -> str:
        """
        Receive al list of tokens and return in one string without stop words 
        and Lemmatized
        """
        tokens = self.clean_text(text)
        # for each token if is not a stop word add the lemma of the word in the list
        lemmas = [t.lemma_ for t in tokens]

        # return the tokens in one string
        return(" ".join(lemmas))

    def stemming(self, text: str) -> str:
        tokens = self.clean_text(text)
        return " ".join([self.stemmer.stem(str(t)) for t in tokens])
    

if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    dataset_name = "cpn120"

    if dataset_name == "cpn27":
        data = pd.read_csv("data/raw_dataset/CPN27.csv", delimiter=",")
    elif dataset_name == "cpn120":
        data = pd.read_csv("data/raw_dataset/CPN120.csv", delimiter=",")

    Cln = TextCleaner("es_core_news_sm", "spanish")
    func = Cln.normalize
    data.iloc[:, 1] = data.iloc[:,1].apply(func)

    if dataset_name == "cpn27":
        data.to_csv("data/raw_dataset/lemmatize/CPN27_lemma.csv", index=False)
    elif dataset_name == "cpn120":
        data.to_csv("data/raw_dataset/normalize/CPN120_normalize.csv", index=False)
