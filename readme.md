#### 7PAM2002-0901-2024 - MSc Data Science Project

Topic - Comparing Data Augmentation Methods – Easy Data Augmentation and Back Translation for text(Intentation) Classification using LSTM.

Research Question - Which data augmentation methods applied on a small dataset outperform models trained without augmentation in terms of accuracy and precision in case of intention(text) classification using LSTM as training models, and by how much do they improve performance?

Supervisor - Dr. Man Lai Tang

Done by - Mohit Agarwal (22031257)


#### About Project
The project investigates data augmentation techniques, namely, easy data augmentation (EDA)
– nosing and back translation – paraphrasing, on a text dialogue-based dataset with intention
as its labels, making it a supervised learning problem. Three models are developed using
long-short term memory (LSTM) architecture, single layer LSTM model, double layer LSTM
model, and convolutional layer stacked on top single LSTM layer, each trained on the dataset
and augmented datasets. The results are compared using accuracy and precision metrics from
each model for all datasets (augmented and non-augmented).
Purpose: To identify which data augmentation method performs well over which model.

#### Dataset Links
[Intent Classification – GitHub](https://github.com/clinc/oos-eval)
[Intent Classification - UCL](https://archive.ics.uci.edu/dataset/570/clinc150)

#### Project Structure 
preprocess_eda.py --- reads the dataset, and has prepocessing class and exploring data class 
lstm.py --- builds the three models, along with hypermodel for tuning 
train_lstm.py ---builds and compiles a model based on given parameters
eda.py ---data augmentaion file for easy data augmentation method
back_translation.py ---data augmentation file for back translation 
main_part_1.ipynb ---calls the files and reads, preprocess and explores the data, finds the hyperparameter and trains model based on them, finally augments the data
main_part_2.ipynb ---reads eda's augmented data, trains them on hyperparameter found on main_part_1.ipynb
main_part_3.ipynb ---reads back translated augmented data, trains them on hyperparameter found on main_part_1.ipynb

Note: The model is saved outside the git directory.
##### Directories
Agument ---stores all the augmented py file and augmented data
Plots ---holds all the plots

#### Required Packages
pandas.2.2.2
matplotlib 
numpy.1.26.4
sklearn.1.5.0
keras.3.6.0 
tensorflow.2.17.0
keras-tuner.1.4.7
wordcloud.1.9.3
nltk.3.8.1
deep-translator.1.11.4

#### Results 
Single LSTM model worked best among the three model on both augmented and non-augmented data. 
In term of precision, the augmentation method, easy data augmentation give better reading for Single-LSTM model, while for f1 score back translation worked better.

#### Future Works 
1. Higher number of epochs can be used to train the single-lstm model.
2. GAN can be used for data augmentation.
3. tf-idf can be used in text vectorization layer.