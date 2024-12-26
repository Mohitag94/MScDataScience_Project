"""This module contains the all the model architecture designs based on LSTM(Long Short Term Memory)
    and a combination with or without Convolutional Layers from Convolutional Neural Networks(CNN), 
    namely, 
    1. Single-LSTM: Only one LSTM layer combined with base layers.
        base layes: it has tokenizing layer and embedding layer.
    2. Stacked_-LSTM: Two stacked LSTM back to back along with the base layers.
    3. Convo-LSTM: One convolution layer followed by base layers, after which a LSTM is added.

The model has a class for tuning hyperparameter in the models mentioned above.
"""

# importing os package
import os
import keras_tuner
import keras

# base path for the fine tuning trails saving
PATH = r"D:\MScDataScience\7.Data_Science_Project\HPO"


class models():
    """creating all three model architectures
    """

    def __init__(self, num_class, vocab_size, embedding_seq_length,
                 textvector_layer, weights=None):
        """initializing the parameters for the model architecture 

            Args:
            num_class: total number of class/traget variable
            vocab_size: vocabulary size for textvectorisation layer
            embedding_seq_length: output length for embedding layer
            textvector_layer: textvectorisation layer
            weights: weights for the embedding layer
        """

        self.num_class = num_class
        self.vocab_size = vocab_size
        self.embedding_seq_length = embedding_seq_length
        self.textvector_layer = textvector_layer
        self.weights = weights

    def base_layer(self):
        """creates base layer, textvectorisation and embedding

        Args:
            weights: weights for the embedding layer
                default to None
        Returns:
            the base model
        """

        # clearing the pervious session of keras
        keras.backend.clear_session()
        model = keras.models.Sequential()
        model.add(self.textvector_layer)
        # add embedding based on weights
        if not self.weights:
            model.add(keras.layers.Embedding(self.vocab_size+2,
                                             self.embedding_seq_length,
                                             trainable="True"))
        else:
            model.add(keras.layers.Embedding(self.vocab_size+2,
                                             self.embedding_seq_length,
                                             trainable="True",
                                             weights=self.weights))
        return model

    def top_layer(self, model):
        """adds top layes, flatten and dense with number of class
        to the model provided

        Args:
            model: keras model to add top layers
        """

        # adding layers flatten and dense for classification
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(self.num_class, activation="softmax"))

    def single_lstm(self, lstm_units, rate, activation):
        """creating a single lstm model

        Args:
            lstm_units: number of units for lstm layer
            rate: dropout and recurrent-dropout rate
            activation: activation used for lstm layer

        Returns:
            single lstm model architecture
        """

        model = self.base_layer()
        # adding lstm layer
        model.add(keras.layers.LSTM(lstm_units,
                                    dropout=rate,
                                    recurrent_dropout=rate,
                                    activation=activation))
        self.top_layer(model=model)

        return model

    def stacked_lstm(self, lstm_units1, lstm_units2, activation,
                     rate1=0.2, rate2=0.2):
        """creating a two stacked lstm model

        Args:
            lstm_units1: number of units for lstm layer1
            lstm_units2: number of units for lstm layer2
            activation: activation used for lstm layers
            rate1: dropout and recurrent-dropout rate 1st lstm layer
                default to 0.2
            rate2: dropout and recurrent-dropout rate 2nd lstm layer
                default to 0.2

        Returns:
            two stacked lstm model architecture
        """

        model = self.base_layer()
        # first lstm layer with retirning sequence
        model.add(keras.layers.LSTM(lstm_units1,
                                    activation=activation,
                                    dropout=rate1,
                                    recurrent_dropout=rate1,
                                    return_sequences=True))
        # second lstm layer
        model.add(keras.layers.LSTM(lstm_units2,
                                    activation=activation,
                                    dropout=rate2,
                                    recurrent_dropout=rate2))
        self.top_layer(model=model)

        return model

    def convo_lstm(self, convo_filters, convo_rate, convo_activation,
                   lstm_units, lstm_rate, lstm_activation, kernel_size=2):
        """creating a convo-lstm model

        Args:
            convo_filters: filter size convolutional layer
            convo_rate: dropout rate for the dropout layer
            convo_activation: activation for convolutional layer
            lstm_units: number of units for lstm layer
            lstm_rate: dropout and recurrent-dropout rate
            lstm_activation: activation used for lstm layer
            kernel_size: the kernel size for convolution layer 
                default to 2

        Returns:
            convo-lstm model architecture
        """

        model = self.base_layer()
        # adding convolutional layer
        model.add(keras.layers.Conv1D(filters=convo_filters,
                                      kernel_size=kernel_size,
                                      activation=convo_activation,
                                      padding="same"))
        model.add(keras.layers.Dropout(convo_rate))
        # adding lstm layer
        model.add(keras.layers.LSTM(lstm_units,
                                    dropout=lstm_rate,
                                    recurrent_dropout=lstm_rate,
                                    activation=lstm_activation))
        self.top_layer(model=model)

        return model


class lstm_hypermodel(keras_tuner.HyperModel):
    """single lstm hypermodel build for hyperparameter optimization
    """

    def __init__(self, num_class, vocab_size, textvector_layer, weights=None):
        """initializing the parameters for the model architecture 

            Args:
                num_class: total number of class/traget variable
                vocab_size: vocabulary size for textvectorisation layer
                embedding_seq_length: output length for embedding layer
                textvector_layer: textvectorisation layer
                weights: weights for the embedding layer
        """

        super().__init__()
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.textvector = textvector_layer
        self.weights = weights

    def build(self, hp):
        """builds the hypermodel for the single lstm model 
        and compiles it

        Args:
            hp: hyperparameter method

        Returns:
            the compiles hypermodel
        """

        # instant of the model
        lstm = models(num_class=self.num_class,
                      vocab_size=self.vocab_size,
                      embedding_seq_length=hp.Int("embedding_seq_length",
                                                  min_value=10,
                                                  max_value=150,
                                                  step=10),
                      textvector_layer=self.textvector,
                      weights=self.weights)

        activations = ["tanh", "sigmoid", "elu", "selu"]

        # creating hypermodel
        model = lstm.single_lstm(lstm_units=hp.Int("lstm_units", min_value=25,
                                                   max_value=200, step=5),
                                 rate=hp.Float("rate", min_value=0.2,
                                               max_value=0.7, step=0.025),
                                 activation=hp.Choice("activation",
                                                      values=activations))

        # learning rate
        lr = hp.Float("learning_rate", min_value=0.001,
                      max_value=0.02, sampling="log")
        # learning optimizer
        optimizer_dict = {
            "SGD": keras.optimizers.SGD(lr),
            "Adam": keras.optimizers.Adam(lr),
            "Nadam": keras.optimizers.Nadam(lr),
            "Adamax": keras.optimizers.Adamax(lr),
            "RMSprop": keras.optimizers.RMSprop(lr)
        }
        # choicing an optimizer
        optimizer = hp.Choice("optimizer",
                              values=["Adam", "SGD", "Nadam",
                                      "Adamax", "RMSprop"])
        # compiling the model
        model.compile(optimizer=optimizer_dict[optimizer],
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=[keras.metrics.Accuracy(),
                               keras.metrics.F1Score(),
                               keras.metrics.Precision()])

        return model

    def fit(self, hp, model, *args, **kwargs):
        """fits the complied hypermodel with shuffle and 
        batch_size as hyperparameter

        Returns: 
            the fit to hypermodel
        """

        return model.fit(*args,
                         shuffle=hp.Boolean("shuffle",
                                            default=True),
                         batch_size=hp.Int("batch_size", min_value=16,
                                           max_value=256, step=16),
                         **kwargs)


class stacked_lstm_hypermodel(keras_tuner.HyperModel):
    """two stacked lstm hypermodel build for hyperparameter optimization
    """

    def __init__(self, num_class, vocab_size, textvector_layer, weights=None):
        """initializing the parameters for the model architecture 

            Args:
                num_class: total number of class/traget variable
                vocab_size: vocabulary size for textvectorisation layer
                embedding_seq_length: output length for embedding layer
                textvector_layer: textvectorisation layer
                weights: weights for the embedding layer
        """

        super().__init__()
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.textvector = textvector_layer
        self.weights = weights

    def build(self, hp):
        """builds the hypermodel for two stacked lstm model 
        and compiles it

        Args:
            hp: hyperparameter method

        Returns:
            the compiles hypermodel
        """

        # instant of the model
        lstm = models(num_class=self.num_class,
                      vocab_size=self.vocab_size,
                      embedding_seq_length=hp.Int("embedding_seq_length",
                                                  min_value=10,
                                                  max_value=150,
                                                  step=10),
                      textvector_layer=self.textvector,
                      weights=self.weights)

        activations = ["tanh", "sigmoid", "elu", "selu"]
        # building the stacked model
        model = lstm.stacked_lstm(lstm_units1=hp.Int("lstm_units1",
                                                     min_value=25,
                                                     max_value=200,
                                                     step=5),
                                  lstm_units2=hp.Int("lstm_units2",
                                                     min_value=25,
                                                     max_value=200,
                                                     step=5),
                                  rate1=hp.Float("rate1",
                                                 min_value=0.2,
                                                 max_value=0.7,
                                                 step=0.025),
                                  rate2=hp.Float("rate2",
                                                 min_value=0.2,
                                                 max_value=0.7,
                                                 step=0.025),
                                  activation=hp.Choice("activation",
                                                       values=activations))
        # learning rates
        lr = hp.Float("learning_rate", min_value=0.001,
                      max_value=0.02, sampling="log")
        # learning optimizer
        optimizer_dict = {
            "SGD": keras.optimizers.SGD(lr),
            "Adam": keras.optimizers.Adam(lr),
            "Nadam": keras.optimizers.Nadam(lr),
            "Adamax": keras.optimizers.Adamax(lr),
            "RMSprop": keras.optimizers.RMSprop(lr)
        }
        # choicing an optimizer
        optimizer = hp.Choice("optimizer",
                              values=["Adam", "SGD", "Nadam",
                                      "Adamax", "RMSprop"])
        # compiling the model
        model.compile(optimizer=optimizer_dict[optimizer],
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=[keras.metrics.Accuracy(),
                               keras.metrics.F1Score(),
                               keras.metrics.Precision()])

        return model

    def fit(self, hp, model, *args, **kwargs):
        """fits the complied hypermodel with shuffle and 
        batch_size as hyperparameter

        Returns: 
            the fit to hypermodel
        """
        return model.fit(*args,
                         shuffle=hp.Boolean("shuffle", default=True),
                         batch_size=hp.Int("batch_size", min_value=16,
                                           max_value=256, step=16),
                         **kwargs)


class convo_lstm_hypermodel(keras_tuner.HyperModel):
    """convolutional-lstm hypermodel build for hyperparameter optimization
    """

    def __init__(self, num_class, vocab_size, textvector_layer, weights=None):
        """initializing the parameters for the model architecture 

            Args:
                num_class: total number of class/traget variable
                vocab_size: vocabulary size for textvectorisation layer
                embedding_seq_length: output length for embedding layer
                textvector_layer: textvectorisation layer
                weights: weights for the embedding layer
        """

        super().__init__()
        self.num_class = num_class
        self.vocab_size = vocab_size
        self.textvector = textvector_layer
        self.weights = weights

    def build(self, hp):
        """builds the hypermodel for convolutional lstm model 
        and compiles it

        Args:
            hp: hyperparameter method

        Returns:
            the compiled hypermodel
        """

        # instant of the model
        lstm = models(num_class=self.num_class,
                      vocab_size=self.vocab_size,
                      embedding_seq_length=hp.Int("embedding_seq_length",
                                                  min_value=10,
                                                  max_value=150,
                                                  step=10),
                      textvector_layer=self.textvector,
                      weights=self.weights)
        # defining the size for convolutional layer
        convo_filters = hp.Int("convo_filters", min_value=32,
                               max_value=512, step=32)
        # defining the kernel size for convolutional layer
        kernel_size = hp.Int("kernel_size", min_value=2,
                             max_value=6, step=1)
        # defining the dropout rate for dropout layer
        convo_rate = hp.Float("convo_rate", min_value=0.2,
                              max_value=0.7, step=0.025)
        # selecting the activation for convolutional layer
        convo_activation = hp.Choice("convo_activation",
                                     values=["relu", "tanh",
                                             "sigmoid", "elu",
                                             "exponential", "selu"])
        # defining the lstm layer's unit
        lstm_units = hp.Int("lstm_units", min_value=25,
                            max_value=200, step=5)
        # defining the lstm layer's dropout and recuurent_dropout
        lstm_rate = hp.Float("lstm_rate", min_value=0.2,
                             max_value=0.7, step=0.025)
        # defining the lstm layer's activation
        lstm_activation = hp.Choice("lstm_activation",
                                    values=["tanh", "sigmoid",
                                            "elu", "selu"])
        # building the hypermodel
        model = lstm.convo_lstm(convo_filters=convo_filters,
                                kernel_size=kernel_size,
                                convo_rate=convo_rate,
                                convo_activation=convo_activation,
                                lstm_units=lstm_units,
                                lstm_rate=lstm_rate,
                                lstm_activation=lstm_activation)
        # learning rate
        lr = hp.Float("learning_rate", min_value=0.001,
                      max_value=0.02, sampling="log")
        # learning optimizer
        optimizer_dict = {
            "SGD": keras.optimizers.SGD(lr),
            "Adam": keras.optimizers.Adam(lr),
            "Nadam": keras.optimizers.Nadam(lr),
            "Adamax": keras.optimizers.Adamax(lr),
            "RMSprop": keras.optimizers.RMSprop(lr)
        }
        # choicing the optimizer
        optimizer = hp.Choice("optimizer",
                              values=["Adam", "SGD", "Nadam",
                                      "Adamax", "RMSprop"])
        # compiling the hypermodel
        model.compile(optimizer=optimizer_dict[optimizer],
                      loss=keras.losses.CategoricalCrossentropy(),
                      metrics=[keras.metrics.Accuracy(),
                               keras.metrics.F1Score(),
                               keras.metrics.Precision()])

        return model

    def fit(self, hp, model, *args, **kwargs):
        """fits the complied hypermodel with shuffle and 
        batch_size as hyperparameter

        Returns: 
            the fit to hypermodel
        """

        return model.fit(*args,
                         shuffle=hp.Boolean("shuffle", default=True),
                         batch_size=hp.Int("batch_size", min_value=16,
                                           max_value=256, step=16),
                         **kwargs)


def fine_tune(x_train, x_val, y_train, y_val,
              num_class, vocab_size, textvector_layer,
              model_name, tuner_name, trials=15, weights=None,
              pre_trials=4, epochs=30, size=3000, path=PATH):
    """fune tuning the hypermodel through either Bayesian or 
    Hyperband for hyperparameter

    Args:
        x_train: training features
        x_val: validation features
        y_train: training classes
        y_val: validation classes
        num_class: total number of class
        vocab_size: vocabulary size for the textvectorization
        textvector_layer: textvector layer
        model_name: hypermodel name to be created
        tuner_name: tuner name for hyperparameter tuning
        trails: number of trails to be runned for tuning
            Note: default to 15
        weights: weights for the embedding layer
            Note: default to None
        pre_trails: each trail to be repeated
            Note: default to 4
        epochs: number of epochs running per trail
            Note: default to 30
        size: number of training records to be used and 1/3rd of size for validation
            Note: default to 3000
        path: the path to save all the trails 
            Note: default to local drive

        Returns:
            tuner for that hypermodel"""

    # hypermodel for single lstm
    if model_name == "single_lstm":
        print("[INFO] Hyperparameter Tuning for Single LSTM...\n")
        hypermodel = lstm_hypermodel(num_class=num_class,
                                     vocab_size=vocab_size,
                                     textvector_layer=textvector_layer,
                                     weights=weights)
        project_name = "single_lstm_model"

    # hypermodel for stacked lstm
    elif model_name == "stacked_lstm":
        print("[INFO] Hyperparameter Tuning for Stacked LSTM...")
        hypermodel = stacked_lstm_hypermodel(num_class=num_class,
                                             vocab_size=vocab_size,
                                             textvector_layer=textvector_layer,
                                             weights=weights)
        project_name = "stacked_lstm_model"

    # hypermodel for convolutional lstm
    elif model_name == "convo_lstm":
        print("[INFO] Hyperparameter Tuning for Convo-LSTM...")
        hypermodel = convo_lstm_hypermodel(num_class=num_class,
                                           vocab_size=vocab_size,
                                           textvector_layer=textvector_layer,
                                           weights=weights)
        project_name = "convo_lstm_model"
    else:
        print("\n*********Wrong Model Selected*********\nOptions are:\n1.'single_lstm'\n2.'stacked_lstm'\n3.'convo_lstm'")
        return

    # for bayesian optimizer
    if tuner_name == "bayesian":
        print("\t[INFO] Hyperparameter Optimizer is Bayesian...")
        tuner = keras_tuner.BayesianOptimization(hypermodel=hypermodel,
                                                 objective="val_accuracy",
                                                 max_trials=trials,
                                                 executions_per_trial=pre_trials,
                                                 directory=os.path.join(
                                                     path, "Bayesian"),
                                                 project_name=project_name,
                                                 overwrite=False)
    #  for hyperband
    elif tuner_name == "hyperband":
        print("\t[INFO] Hyperparameter Optimizer is Hyperband...")
        tuner = keras_tuner.Hyperband(hypermodel=hypermodel,
                                      objective="val_accuracy",
                                      hyperband_iterations=trials,
                                      directory=os.path.join(
                                          path, "Hyperband"),
                                      project_name=project_name,
                                      overwrite=False)
    else:
        print("\n*********Wrong Optimizer Selected*********\nOptions are:\n1.'bayesian'\n2.'hyperband'")
        return
    print("\n", tuner.search_space_summary())

    # funing tuning the given set of data with selected hypermodel
    tuner.search(x_train.iloc[:size], y_train[:size, :],
                 epochs=epochs,
                 validation_data=(x_val.iloc[:int(size/3)],
                                  y_val[:int(size/3), ]))

    print(tuner.results_summary())

    return tuner
# End-of-file (EOF)
