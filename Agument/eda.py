"""Method-1 - Easy Data Augmentation(EDA) for Data Augmentation for Text Classification Project

EDA algorithm has four methods, 
    1. Random Synonyms(SR): Randomly select n non-stopwords from the text 
                             and replace them with their synonyms.
    2. Random Insert(RI): Randomly select n non-stopwords from the text and 
                         insert their synonyms at random positions inside the text.
    3. Random Swap(RS): Randomly select n pairs of words and swap their positions.
    4. Random Delection(RD): Delects a word(s) from the text having lower probability p.

one method is randomly selected to augment the data, n times, where n is user defined.
"""

# importing packages...
import random
import os
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords

# defining the path to save the back translated file
PATH = r"D:\MScDataScience\7.Data_Science_Project\SourceCode\Agument\Augment_Data\EDA"
# defining english stopwords from nltk corpus
STOPSWORD_ENG = set(stopwords.words("english"))


class eda():
    """performs the easy data augmentation's four methods, 
    "SR", "RI", "RD", "RS".
    """

    def find_nonstopwords_indices(self, text):
        """finds the indices of non-stopwords from the text

        Args:
            text: the text of which non-stopwords indices to be located

        Returns:
            the list of non-stopwords positions
        """

        indices = []
        for index, word in enumerate(text):
            # locating the non-stopwords position
            if word.lower() not in STOPSWORD_ENG:
                indices.append(index)
        return indices

    def find_synonyms(self, word):
        """finds the synonyms from the nltk wordnet corpus

        Args:
            word: the word for which synonyms needs to be found

        Returns:
            the list of synonyms
        """

        synonyms = []
        # synsets contains set of synonyms
        for synset in wordnet.synsets(word):
            # getting all words with same meaning to the words
            for lemma in synset.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())

        return synonyms

    def eda_SR(self, original_text, n):
        """for each n randomly chosen non-stopword, it is swapped with 
        its respective synonyms randomly selected from the list of synonyms
        of that word. 

        Args:
            original_text: the original string which needs augmented
            n: number of words to be replaced with synonyms

        Returns:
            the augmented string/text
        """

        text = original_text.split().copy()
        # getting the non-stopwords positons
        text_nonstopwords_indices = self.find_nonstopwords_indices(text=text)

        # checking if the text only have stopwords
        if not text_nonstopwords_indices:
            return original_text

        # setting n to random number incase the n is larger than non-stopwords
        if n > len(text_nonstopwords_indices):
            n = random.randint(0,
                               int(len(text_nonstopwords_indices)/2))

        # replacing the non-stopwords through randomly chosing words
        for _ in range(n):
            index = random.choice(text_nonstopwords_indices)
            # removing the selected word, so it's not reselected
            text_nonstopwords_indices.remove(index)
            synonyms = self.find_synonyms(text[index])
            if not synonyms:
                continue
            # selecting a random synonyms for that word
            text[index] = random.choice(synonyms).replace("_", " ")
            if not text_nonstopwords_indices:
                break

        return " ".join(text)

    def eda_RI(self, original_text, n):
        """for each n randomly chosen non-stopword, it's respective synonyms is 
        randomly selected from the list of synonyms of that word, and randomly 
        inserted into any positon. 

        Args:
            original_text: the original string which needs augmented
            n: number of words to be replaced with synonyms

        Returns:
            the augmented string/text
        """

        text = original_text.split().copy()
        # getting the non-stopwords positons
        text_nonstopwords_indices = self.find_nonstopwords_indices(text=text)

        # checking if the text only have stopwords
        if not text_nonstopwords_indices:
            return original_text

        # setting n to random number incase the n is larger than non-stopwords
        if n > len(text_nonstopwords_indices):
            n = random.randint(0,
                               int(len(text_nonstopwords_indices)/2))

        # inserting a synonyms
        for _ in range(n):
            index = random.choice(text_nonstopwords_indices)
            # removing the selected word, so it's not reselected
            text_nonstopwords_indices.remove(index)
            synonyms = self.find_synonyms(text[index])
            if not synonyms:
                continue
            # random synonyms is inserted at random index
            text.insert(random.randint(0, len(text)-1),
                        random.choice(synonyms).replace("_", " "))
            if not text_nonstopwords_indices:
                break

        return " ".join(text)

    def eda_RD(self, original_text, p):
        """for each n randomly chosen word from the text, a random 
        probability is generated and checked with the given probability, p, 
        if its less then, that word is deleted.

        Args:
            original_text: the original string which needs augmented
            p: probability for the string

        Returns:
            the augmented string/text
        """

        text = original_text.split().copy()

        if (p <= 0 or p >= 1) or len(text) <= 1:
            return original_text

        for word in text:
            # generating probability for a word
            word_probability = random.random()
            if word_probability <= p:
                # removing the word
                text.remove(word)

        if not text:
            return original_text

        return " ".join(text)

    def eda_RS(self, original_text, n):
        """for each n randomly chosen pairs of words,
        their position is swapped.

        The pairs are non repeat.

        Args:
            original_text: the original string which needs augmented
            n: number of pair of words to be swapped

        Returns:
            the augmented string/text
        """

        text = original_text.split().copy()
        text_length = len(text)
        index = [i for i in range(text_length)]
        while (len(index) > 1 and n > 0):
            n -= 1
            # chosing the first word
            first_index = random.choice(index)
            index.remove(first_index)
            # chosing the second word
            second_index = random.choice(index)
            index.remove(second_index)
            # swapping the chosen pair's position
            text[first_index], text[second_index] = text[second_index], text[first_index]

        return " ".join(text)


class augment_data(eda):
    """augmenation is performed based on EDA's algorithm on the given data
    """

    def __init__(self, data, augment_sizes, path=PATH):
        """initializing the parameters for the augmentation

         Args:
            data: the original data on which augmentation is done
            augment_sizes: list of augmentation size
            path: path the augmented data to be saved
                    Note: default to Augment directory
        """

        super().__init__()
        self.augment_sizes = augment_sizes
        self.original_data = data
        # lsiting all the EDA methods
        self.eda_methods = ["SR", "RI", "RD", "RS"]
        self.augment_path = path

    def augment(self):
        """from the methods, "SR", "RI", "RD", "RS", 
        one method is selected randomly for each string from 
        the list of strings

        Returns:
            saves the augmented data to the given path
        """

        # extracting text and class from the data
        queries = self.original_data.iloc[:, 0].copy()
        labels = self.original_data.iloc[:, 1].copy()

        for size in self.augment_sizes:
            print(f"[INFO] Augmenting data of size {size}...")
            augment_queries = []
            # note - not augmenting labels, just to keep track
            augment_labels = []
            # augmenting data for each records
            for i, query in enumerate(queries):
                for _ in range(size):
                    augment_labels.append(labels[i])
                    # randomly selecting one method
                    method = random.choice(self.eda_methods)
                    n = random.randint(0, len(query)-1)
                    if method == "SR":
                        augment_queries.append(super().eda_SR(query, n))
                    elif method == "RI":
                        augment_queries.append(super().eda_RI(query, n))
                    elif method == "RD":
                        augment_queries.append(
                            super().eda_RD(query, random.random()))
                    elif method == "RS":
                        augment_queries.append(super().eda_RS(query, n))
            print("\t Augmented.")
            print("[INFO] Saving the augmented data to disk...")
            pd.DataFrame(
                {"Query": augment_queries,
                 "Intent": augment_labels}).\
                to_csv(os.path.join(self.augment_path,
                                    f"eda_augmented_data_size_{size}.csv"))
            print("\t Saved.\n")
