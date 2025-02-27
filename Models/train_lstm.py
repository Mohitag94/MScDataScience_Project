"""This module contents functions to build the single lstm, stacked lstm
and convo-lstm models, and a function to plot accuracy, loss, f1-score
and precesion graphs for validation and training."""


# importing os package
import os
# import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import lstm


def single_lstm_model(embedding_seq_length,
                      lstm_units,
                      rate,
                      activation,
                      optimizer,
                      lr,
                      num_class,
                      vocab_size,
                      textvector_layer):
    """builds the single layer lstm model with
    given parameter

    Args:
        embedding_seq_length: output lenth for embedding layer
        lstm_units: units for lstm layer
        rate: droprate
        activation: activation fucntion
        optimizer: optimizer for network
            options: "SGD", "Adam", "Nadam", "Adamax", "RMSprop"
        lr: learning rate for network
        num_class: number of class
        vocab_size: vocabulary size for embedding layer
        textvector_layer: textvectorization layer

    Returns:
        model: the complied model
    """

    # optimizer algorithms dict
    optimizer_dict = {
        "SGD": keras.optimizers.SGD(learning_rate=lr),
        "Adam": keras.optimizers.Adam(learning_rate=lr),
        "Nadam": keras.optimizers.Nadam(learning_rate=lr),
        "Adamax": keras.optimizers.Adamax(learning_rate=lr),
        "RMSprop": keras.optimizers.RMSprop(learning_rate=lr)
    }

    # object for the lstm models
    models = lstm.models(num_class=num_class,
                         vocab_size=vocab_size,
                         embedding_seq_length=embedding_seq_length,
                         textvector_layer=textvector_layer)

    # single lstm model with given parameter
    model = models.single_lstm(lstm_units=lstm_units,
                               rate=rate,
                               activation=activation)
    # single lstm model compiling
    model.compile(optimizer=optimizer_dict[optimizer],
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.Accuracy(),
                           keras.metrics.F1Score(threshold=0.5),
                           keras.metrics.Precision()])

    return model


def stacked_lstm_model(embedding_seq_length,
                       lstm_units1,
                       lstm_units2,
                       rate1,
                       rate2,
                       activation,
                       optimizer,
                       lr,
                       num_class,
                       vocab_size,
                       textvector_layer):
    """builds the stacked layer lstm model with
    given parameter

    Args:
        embedding_seq_length: output lenth for embedding layer
        lstm_units1: units for lstm layer 1
        lstm_units2: units for lstm layer 2
        rate1: droprate for layer 1
        rate2: droprate for layer 2
        activation: activation fucntion
        optimizer: optimizer for network
            options: "SGD", "Adam", "Nadam", "Adamax", "RMSprop"
        lr: learning rate for network
        num_class: number of class
        vocab_size: vocabulary size for embedding layer
        textvector_layer: textvectorization layer

    Returns:
        model: the complied model
    """

    # optimizer algorithms dict
    optimizer_dict = {
        "SGD": keras.optimizers.SGD(learning_rate=lr),
        "Adam": keras.optimizers.Adam(learning_rate=lr),
        "Nadam": keras.optimizers.Nadam(learning_rate=lr),
        "Adamax": keras.optimizers.Adamax(learning_rate=lr),
        "RMSprop": keras.optimizers.RMSprop(learning_rate=lr)
    }

    # object for the lstm models
    models = lstm.models(num_class=num_class,
                         vocab_size=vocab_size,
                         embedding_seq_length=embedding_seq_length,
                         textvector_layer=textvector_layer)

    # single lstm model with given parameter
    model = models.stacked_lstm(lstm_units1=lstm_units1,
                                lstm_units2=lstm_units2,
                                rate1=rate1,
                                rate2=rate2,
                                activation=activation)
    # single lstm model compiling
    model.compile(optimizer=optimizer_dict[optimizer],
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.Accuracy(),
                           keras.metrics.F1Score(threshold=0.5),
                           keras.metrics.Precision()])

    return model


def convo_lstm_model(embedding_seq_length,
                     convo_filters,
                     convo_rate,
                     convo_activation,
                     lstm_units,
                     lstm_rate,
                     lstm_activation,
                     optimizer,
                     lr,
                     num_class,
                     vocab_size,
                     textvector_layer,
                     kernel_size=2):
    """builds the convo-lstm model with
    given parameter

    Args:
        embedding_seq_length: output lenth for embedding layer
        lstm_units1: units for lstm layer 1
        lstm_units2: units for lstm layer 2
        rate1: droprate for layer 1
        rate2: droprate for layer 2
        activation: activation fucntion
        optimizer: optimizer for network
            options: "SGD", "Adam", "Nadam", "Adamax", "RMSprop"
        lr: learning rate for network
        num_class: number of class
        vocab_size: vocabulary size for embedding layer
        textvector_layer: textvectorization layer

    Returns:
        model: the complied model
    """

    # optimizer algorithms dict
    optimizer_dict = {
        "SGD": keras.optimizers.SGD(learning_rate=lr),
        "Adam": keras.optimizers.Adam(learning_rate=lr),
        "Nadam": keras.optimizers.Nadam(learning_rate=lr),
        "Adamax": keras.optimizers.Adamax(learning_rate=lr),
        "RMSprop": keras.optimizers.RMSprop(learning_rate=lr)
    }

    # object for the lstm models
    models = lstm.models(num_class=num_class,
                         vocab_size=vocab_size,
                         embedding_seq_length=embedding_seq_length,
                         textvector_layer=textvector_layer)

    # single lstm model with given parameter
    model = models.convo_lstm(convo_filters=convo_filters,
                              convo_rate=convo_rate,
                              convo_activation=convo_activation,
                              kernel_size=kernel_size,
                              lstm_units=lstm_units,
                              lstm_rate=lstm_rate,
                              lstm_activation=lstm_activation)

    # single lstm model compiling
    model.compile(optimizer=optimizer_dict[optimizer],
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.Accuracy(),
                           keras.metrics.F1Score(threshold=0.5),
                           keras.metrics.Precision()])

    return model


def model_history(x, y,
                  x_val, y_val,
                  model,
                  batch_size,
                  epochs,
                  path,
                  filename):
    """trains the given model with validation set, and 
    saves the models and csv log file to the provided location

    Args:
        x: non-label training data
        y: label training data
        x_val: non-label validation data
        y_val: label validation data
        model: model to train
        batch_size : batch size training
        epochs: number of epochs to be runned
        path: location for the saving files
        filename: filename for checkpoints and csvlogger

    Returns:
        history: the training history
    """

    # checkpoints
    checkpoint = keras.callbacks.ModelCheckpoint(filepath=os.path.join(path,
                                                                          f"{filename}.keras"),
                                                    monitor="val_accuracy",
                                                    mode="max",
                                                    save_best_only=True)
    # csvlogger
    csvlog = keras.callbacks.CSVLogger(filename=os.path.join(path,
                                                                f"{filename}.csv"),
                                          separator=",",
                                          append=False)
    # fitting the model to the data
    history = model.fit(x=x, y=y,
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint, csvlog],
                        verbose=1)

    return history


def plot(history, title, path):
    """plots line graphs for a trained model and 
    saves them in the given location"""

    # metrics used
    metrics = ["accuracy", "loss", "precision"]

    # looping over metrics and plotting each one
    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history[f"val_{metric}"])
        plt.legend(["Training", "Validation"])
        plt.title(f"{title} - {metric.capitalize()}")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.savefig(os.path.join(path, f"{title}-{metric}.png"))
        plt.show()
        plt.clf()
# End-of-file (EOF)
