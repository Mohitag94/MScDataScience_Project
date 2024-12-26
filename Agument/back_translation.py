"""Method-2 - Back Translation for Data Augmentation in Text Classification Project

Back Translation is a technique used to increase the amount of data via 
translating the text to an intermediate language and back to the original language.

The original language is set to English, as the data for this project is in English.

"""


# importing packages...
import os
import pandas as pd
from deep_translator import GoogleTranslator

# defining the path to save the back translated file
PATH = r"D:\MScDataScience\7.Data_Science_Project\SourceCode\Agument\Augment_Data\Back_Translation"

# selecting intermediate lanuguages
traget_languages = ["hi", "fr", "de", "es"]


class back_translate():
    """back translates the list of given text using GOOGLE tanslator from 
    deep_translator package, either with using Hindi, French, Germen and Spanish
    or with a list of languages.
    """

    def __init__(self, data, tragets=traget_languages, path=PATH):
        """initializing the parameters for the augmentation

         Args:
            data: the original data on which augmentation is done
            traget_languages: the list of intermediate languages
                Note: default to hindi, french, germen, spanish
            path: path the augmented data to be saved
                Note: default to Augment directory
        """

        self.traget_languages = tragets
        self.path = path
        self.data = data

    def google_translate(self, original_text, traget):
        """back translates the texts at once using GOOGLE translator 

        Args:
            original_test: the intended text to be translated 
            traget: the language intended for text translation

        Returns: 
            The back translated and intermediate translated text
        """

        # translating for the given lang
        translated_text = GoogleTranslator(source="en",
                                           target=traget).translate_batch(original_text)

        return GoogleTranslator(source=traget, target="en").translate_batch(translated_text)

    def augment(self):
        """
        back translates the texts at once using GOOGLE translator 

        Args:
            original_test: the intended text to be translated 
            traget: the language intended for text translation

        Returns: 
            The back translated to texts 
        """

        # extracting text and classes from data
        original_text = self.data.iloc[:, 0].to_list()
        labels = self.data.iloc[:, 1]
        # augmenting for each language
        for traget in self.traget_languages:
            print(f"[INFO] Augmenting data using language {traget} ...")
            # back translating the data
            back_translated_text, \
                intermediate_tranlated_text = self.google_translate(
                    original_text, traget)
            # saving the translated and back translated data
            pd.DataFrame({
                "Original Query": original_text,
                f"Intermediate Query {traget}": intermediate_tranlated_text,
                "Back Translated Query": back_translated_text,
                "Intent": labels
            }).to_csv(os.path.join(self.path,
                                   f"back_translated_augmented_data_with_intermediate_lang_{traget}.csv"))
            # saving the back translated data
            pd.DataFrame({
                "Query": back_translated_text,
                "Intent": labels
            }).to_csv(os.path.join(self.path,
                                   f"back_translated_augment_data_{traget}.csv"))
            print("\t Augmented and saved to the disk.")
