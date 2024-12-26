"""This module preprocesses the data, lower the text, removes the 
alphanumerical words and punctuations, stopwords and finally lemmetizes 
the remaining words. The traget or classes are one-hot encoded to prevent 
any form of ordering among them.

Also the module has various exploratory data analysis(eda) functions, 
for counting characters in text per record, number of number words per record,
average length of word per record.

The module has also the read function to read the data from the local drive.
"""


# importing packages...
from sklearn.preprocessing import OneHotEncoder
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import contractions
import json
import string
import re
import os

# nltk packages...
import nltk
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# downloading stopwords
nltk.download("stopwords")

# setting stopwords to English as the data is in eng.
STOPSWORD_ENG = set(stopwords.words("english"))
# creating a lemmatizer instant
lemmatizer = WordNetLemmatizer()

# setiing plot style
plt.style.use("seaborn-v0_8-darkgrid")

# path to data
DATA_PATH = r"D:\MScDataScience\7.Data_Science_Project\SourceCode\clinc150\clinc150\data_small.json"

# base path for plots storage
PATH = r"D:\MScDataScience\7.Data_Science_Project\SourceCode\Plots"

# initialising the encoder
ohe = OneHotEncoder(handle_unknown="ignore")


def read_data(path=DATA_PATH):
    """function reads the json file and converts to dataframe

    Args:
        path: location to the json file
            deafult to data path in drive

    Returns:
        train_data: training set dataframe
        val_data: validation set dataframe
        test_data: testing set dataframe
        class_length: number of classes in the data
    """

    # loading the data
    with open(path) as data:
        clinc150_small = json.load(data)
    # loading training, validation and testing sets from the file..
    # training data
    train_data = pd.DataFrame(clinc150_small["train"],
                              columns=["Query", "Intent"])
    # validation data
    val_data = pd.DataFrame(clinc150_small["val"],
                            columns=["Query", "Intent"])
    # testing data
    test_data = pd.DataFrame(clinc150_small["test"],
                             columns=["Query", "Intent"])
    # number of classes
    class_length = len(train_df.iloc[:, 1].unique())

    return train_data, val_data, test_data, class_length


class pre_process():
    """preprocessing the data including encoding the tragets
    """

    def __init__(self, data):
        """initializing the parameters for the augmentation

         Args:
            data: data to be preprocessed
        """

        # spilting the data into features and tragets
        self.x_data = data.iloc[:, 0]
        self.y_data = data[["Intent"]]

    def preprocess(self):
        """preprocessing the data, lowering, expanding contraction,
        removing alphanumerical words and punctuation, and stopwords 
        removable

        Returns:
            the preprocessed data
        """

        # lowing the query
        self.x_data = self.x_data.apply(lambda query: query.lower())
        # expanding the contractions
        self.x_data = self.x_data.apply(lambda query: contractions.fix(query))
        # removing the digits and alphanumerical words
        self.x_data = self.x_data.apply(
            lambda query: re.sub(r"\w*\d\w*", "", query))
        # removing punctuation from
        self.x_data = self.x_data.apply(lambda query:
                                        query.translate(str.maketrans("",
                                                                      "",
                                                                      string.punctuation)))
        # removing english stopwords
        self.x_data = self.x_data.apply(lambda query:
                                        " ".join(word for word in
                                                 query.split() if word not in STOPSWORD_ENG))

        return self.x_data

    def lemmatise(self):
        """the words are lemmatized

        Returns:
            the lemmatized data
        """

        self.x_data = self.x_data.apply(lambda query:
                                        " ".join(lemmatizer.lemmatize(word)
                                                 for word in query.split()))
        return self.x_data

    def encode_class(self):
        return ohe.transform(self.y_data).toarray()

    # def domain_map(self, domain=domians):


class eda():
    """exploring the data"""

    def __init__(self, data, path=PATH):
        """initializing the parameters for the augmentation

         Args:
            data: data to be explored
            path: path to save plots 
                default to local Plots directory
        """

        self.data = data
        self.path = path

    def null_check(self, title):
        """checking null in the data

        Args:
            title: title for the plot

        Returns:
            saves the null plot to the drive
        """

        ax = self.data.isna().sum().plot(kind="bar",
                                         title=title,
                                         xlabel="Columns",
                                         ylabel="No. of Null Values")
        ax.figure.savefig(os.path.join(self.path, f"null_plot_{title}.png"))
        plt.show()

    def query_per_class(self):
        """checking number of records per class

        Returns:
            count aggregated per class dataframe 
        """

        return self.data.groupby(by="Intent").agg({"Query": "count"})

    def char_per_query(self, title):
        """calculates the number of character per records including whitespaces

        Args:
            title: title for the plot

        Returns:
            saves the plot to the drive
        """

        # calculating num of char per record
        char_per_query_df = self.data["Query"].str.len()
        print("[INFO] Minimum Number of Charaters in a query is: ",
              char_per_query_df.min())
        print("[INFO] Maximum Number of Charaters in a query is: ",
              char_per_query_df.max())
        # ploting and saving the graph
        ax = char_per_query_df.plot(kind="hist",
                                    title=f"Character per queries - {title}",
                                    ylabel="Queries",
                                    xlabel="No. of Charaters")
        ax.figure.savefig(os.path.join(
            self.path, f"charater_per_query_{title}.png"))
        plt.show()

    def word_per_query(self, title):
        """calculates the number of words per records 

        Args:
            title: title for the plot

        Returns:
            saves the plot to the drive
        """

        # tokenizing the text and counting the num of words
        word_per_query_df = self.data["Query"].str.split().str.len()
        print("[INFO] Minimum Number of Words in a query is: ",
              word_per_query_df.min())
        print("[INFO] Maximum Number of Words in a query is: ",
              word_per_query_df.max())
        # ploting and saving
        ax = word_per_query_df.plot(kind="hist",
                                    title=f"Words per queries - {title}",
                                    ylabel="Queries",
                                    xlabel="No. of Words")
        ax.figure.savefig(os.path.join(
            self.path, f"word_per_query_{title}.png"))
        plt.show()

    def avg_word_len_per_query(self, title):
        """calculates the number of words per records 

        Args:
            title: title for the plot

        Returns:
            saves the plot to the drive
        """

        # tokenizing and getting average word length of word per query
        avg_word_len_per_query = self.data["Query"].str.split().\
            apply(lambda words: [len(word) for word in words]).\
            map(lambda avg_word: np.mean(avg_word))
        print("[INFO] Minimum Number of Average Words Length in a query is: ",
              avg_word_len_per_query.min())
        print("[INFO] Maximum Number of Average Words Length in a query is: ",
              avg_word_len_per_query.max())
        # ploting and saving
        ax = avg_word_len_per_query.plot(kind="hist",
                                         title=f"Average Word length per queries - {
                                             title}",
                                         ylabel="Queries",
                                         xlabel="Average Word Length")
        ax.figure.savefig(os.path.join(
            self.path, f"avg_word_per_query_{title}.png"))

        plt.show()

    def word_freq_per_class(self):
        """counts the frequency of words per class

        Returns:
            nltk frequency dict for all class
        """

        # checking for word_frequency directory, if not exists then create
        if not os.path.exists(os.path.join(self.path, "Word_Frequency")):
            os.mkdir(os.path.join(self.path, "Word_Frequency"))

        # word frequency dict for all class
        word_freqs_all_class = {}

        # accumulating all the record per class
        data = self.data.groupby(by="Intent").agg({"Query": "sum"})
        # creating frequency dict and ploting top 20 per class
        for i, index in enumerate(data.index):
            word_freq = FreqDist(word_tokenize(data.iloc[i, 0]))
            # apending the frequency to dict
            word_freqs_all_class[index] = word_freq
            ax = word_freq.plot(20, cumulative=False, show=False)
            ax.set_xlabel("Words")
            ax.set_title(f"20 Highest Words Frequency for {index}")
            ax.legend(["Counts"])
            ax.figure.savefig(os.path.join(self.path,
                                           f"Word_Frequency/{index}_20_highest_words_before_preprocess.png"))

        return word_freqs_all_class

    def word_cloud_per_class(self):
        """counts the frequency of words per class

        Returns:
            nltk frequency dict for all class
        """

        # checking for word_cloud directory, if not exists then create
        if not os.path.exists(os.path.join(self.path, "Word_Cloud")):
            os.mkdir(os.path.join(self.path, "Word_Cloud"))

        # word frequency dict for all class
        word_cloud_all_class = {}

        # accumulating all the record per class
        data = self.data.groupby(by="Intent").agg({"Query": "sum"})
        # creating word_could and ploting top 20 per class
        for i, index in enumerate(data.index):
            word_cloud =
# one function with number of records per set


# getting the data
train_df, val_df, test_df, num_intent = read_data()

# fitting the encoder to the train's traget
ohe.fit(train_df.loc["Intent"])
