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
preprocess_eda.py --- reads the dataset, and has prepocessing class and exploring data class </br>
lstm.py --- builds the three models, along with hypermodel for tuning </br>
train_lstm.py ---builds and compiles a model based on given parameters </br>
eda.py ---data augmentaion file for easy data augmentation method </br>
back_translation.py ---data augmentation file for back translation </br>
main_part_1.ipynb ---calls the files and reads, preprocess and explores the data, finds the hyperparameter and trains model based on them, finally augments the data </br>
main_part_2.ipynb ---reads eda's augmented data, trains them on hyperparameter found on main_part_1.ipynb </br>
main_part_3.ipynb ---reads back translated augmented data, trains them on hyperparameter found on main_part_1.ipynb </br>

Note: The model is saved outside the git directory.</br>

##### Directories
Agument ---stores all the augmented py file and augmented data </br>
Plots ---holds all the plots </br>

#### Required Packages
pandas.2.2.2 </br>
matplotlib </br>
numpy.1.26.4 </br>
sklearn.1.5.0 </br>
keras.3.6.0 </br>
tensorflow.2.17.0 </br>
keras-tuner.1.4.7 </br>
wordcloud.1.9.3 </br>
nltk.3.8.1 </br> 
deep-translator.1.11.4 </br>

#### Results 
Single LSTM model worked best among the three model on both augmented and non-augmented data. </br>
In term of precision, the augmentation method, easy data augmentation give better reading for Single-LSTM model, while for f1 score back translation worked better. </br>

#### Future Works 
1. Higher number of epochs can be used to train the single-lstm model. </br>
2. GAN can be used for data augmentation. </br>
3. tf-idf can be used in text vectorization layer. </br>
