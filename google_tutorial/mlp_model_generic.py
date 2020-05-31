from utils import metric
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from imblearn.over_sampling import SMOTE
import typing
import numpy as np


def ngram_vectorize(train_texts, train_labels, val_texts):
    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': (1, 2),  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': 'word',  # Split text into word tokens.
            'min_df': 2,
            }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Limit on the number of features. We use the top 20K features.
    TOP_K = 20000

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val, selector, vectorizer


def mlp_model(layers, units, dropout_rate, input_shape, num_classes, learning_rate, output_bias=None):
    """Creates an instance of a multi-layer perceptron model.

    # Arguments
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of the layers.
        dropout_rate: float, percentage of input to drop at Dropout layers.
        input_shape: tuple, shape of input to the model.
        num_classes: int, number of output classes.

    # Returns
        An MLP model instance.
    """
    if output_bias is not None:
        print("Adding output_bias")
        output_bias = tf.keras.initializers.Constant(output_bias)

    op_units, op_activation = metric.get_last_layer_units_and_activation(num_classes)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dropout(rate=dropout_rate, input_shape=input_shape))
    # model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=input_shape))

    for _ in range(layers-1):
        model.add(tf.keras.layers.Dense(units=units, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))

    print("output_bias", output_bias)
    model.add(tf.keras.layers.Dense(units=op_units, activation=op_activation, bias_initializer=output_bias))

    METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'),
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc'),
              ]

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

    return model


def get_class_weight(train_labels):
    not_asshole = np.count_nonzero(train_labels == 0)
    asshole = np.count_nonzero(train_labels)
    print("There is ", asshole, " asshole and ", not_asshole, " not asshole")
    total = not_asshole + asshole

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_not_asshole = (1 / not_asshole)*(total)/2.0
    weight_for_assole = (1 / asshole)*(total)/2.0

    class_weight = {0: weight_for_not_asshole, 1: weight_for_assole}

    print('Weight for class 0: {:.2f}'.format(weight_for_not_asshole))
    print('Weight for class 1: {:.2f}'.format(weight_for_assole))
    return class_weight


def train_ngram_model(data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2,
                      use_class_weight=True):
    """Trains n-gram model on the given dataset.

    # Arguments
        data: tuples of training and test texts and labels.
        learning_rate: float, learning rate for training model.
        epochs: int, number of epochs.
        batch_size: int, number of samples per batch.
        layers: int, number of `Dense` layers in the model.
        units: int, output dimension of Dense layers in the model.
        dropout_rate: float: percentage of input to drop at Dropout layers.

    # Raises
        ValueError: If validation data has label values which were not seen
            in the training data.
    """
    # Get the data.
    (x_train, train_labels), (x_val, val_labels) = data

    # print("train size", x_train.shape)
    # print("train size", train_labels.shape)
    # print("train size", x_val.shape)
    # print("train size", val_labels.shape)

    # Verify that validation labels are in the same range as training labels.
    num_classes = metric.get_num_classes(train_labels)
    print("Num classes:", num_classes)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
                             unexpected_labels=unexpected_labels))


    class_weights = get_class_weight(train_labels)

    # print("text", x_train)
    # Vectorize texts.
    # x_train, x_val = ngram_vectorize(
    # train_texts, train_labels, val_texts)

    # x_train = x_train.todense()
    # x_val = x_val.todense()

    # if smote == True:
    #     print("Smoting")
    #     x_train, train_labels = smote_f(x_train, train_labels)

    # Create model instance.
    # x_train.shape[1:]

    model = mlp_model(layers=layers,
                      units=units,
                      dropout_rate=dropout_rate,
                      input_shape=x_train.shape[1:],
                      num_classes=num_classes)

    # model.load_weights(initial_weights)

    METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'),
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc'),
              ]

    # Compile model with learning parameters.
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

    # Create callback for early stopping on validation loss. If the loss does
    # not decrease in two consecutive tries, stop training.
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)]

    # Train and validate model.
    history = model.fit(
            x=x_train,
            y=train_labels,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(x_val, val_labels),
            verbose=1,  # Logs once per epoch.
            batch_size=batch_size,
            class_weight=class_weights)

    # Print results.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # Save model.
    model.save('AITA_mlp_model.h5')
    # return history['val_acc'][-1], history['val_loss'][-1]
    return history
