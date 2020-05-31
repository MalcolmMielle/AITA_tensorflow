# import sys
import os
from google_tutorial import mlp_model_generic
from utils import get_data
import numpy as np
import tensorflow as tf
from utils import metric
import matplotlib as mpl
import matplotlib.pyplot as plt

import tempfile

import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix

# %% [markdown]
#
# Let's load/save the processed data.
#

# %% codecell
# script_path = os.path.dirname(__file__)
lightweight = True
if not lightweight:
    # PICKLE = "./google_tutorial_bag_of_word.pkl"
    DATA_PATH = "../data/aita_clean.csv"
    FILE_TRAIN = "google_tutorial/saved_data/train_google.npy"
    FILE_TRAIN_LABELS = "google_tutorial/saved_data/train_google_labels.npy"
    FILE_VALIDATION = "google_tutorial/saved_data/validation_google.npy"
    FILE_VALIDATION_LABELS = "google_tutorial/saved_data/validation_google_labels.npy"
    FILE_TEST = "google_tutorial/saved_data/test_google.npy"
    FILE_TEST_LABELS = "saved_data/test_google_labels.npy"
else:
    # PICKLE = "./google_tutorial_bag_of_word.pkl"
    DATA_PATH = "../data/aita_clean_light.csv"
    FILE_TRAIN = "google_tutorial/saved_data/train_google_light.npy"
    FILE_TRAIN_LABELS = "google_tutorial/saved_data/train_google_labels_light.npy"
    FILE_VALIDATION = "google_tutorial/saved_data/validation_google_light.npy"
    FILE_VALIDATION_LABELS = "google_tutorial/saved_data/validation_google_labels_light.npy"
    FILE_TEST = "google_tutorial/saved_data/test_google_light.npy"
    FILE_TEST_LABELS = "google_tutorial/saved_data/test_google_labels_light.npy"

# %% codecell
if not os.path.exists(FILE_TRAIN):
    print("Crate DATA")
    # Generate data
    data = get_data.generating_data(file_path=DATA_PATH)
    train, validate, test = np.split(data.sample(frac=1), [int(.6 * len(data)), int(.8 * len(data))])

    traindataset = train.text.to_numpy()
    trainlabels = train.is_asshole.to_numpy()
    validatedataset = validate.text.to_numpy()
    validatelabels = validate.is_asshole.to_numpy()
    testdataset = test.text.to_numpy()
    testlabels = test.is_asshole.to_numpy()

    x_train, x_val, selector, vectorizer = mlp_model_generic.ngram_vectorize(traindataset, trainlabels, validatedataset)
    x_train = x_train.todense()
    x_val = x_val.todense()

    x_test = vectorizer.transform(testdataset)
    x_test = selector.transform(x_test).astype('float32')
    x_test = x_test.todense()

    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    smote = False

    if smote is True:
        print("Smoting")
        x_train, trainlabels = mlp_model_generic.smote_f(x_train, trainlabels)

    print(x_train.shape)
    print(x_val.shape)

    np.save(FILE_TRAIN, x_train)
    np.save(FILE_TRAIN_LABELS, trainlabels)
    np.save(FILE_VALIDATION, x_val)
    np.save(FILE_VALIDATION_LABELS, validatelabels)
    np.save(FILE_TEST, x_test)
    np.save(FILE_TEST_LABELS, testlabels)

else:
    print("LOAD PAST DATA")
    x_train = np.load(FILE_TRAIN, allow_pickle=True)
    trainlabels = np.load(FILE_TRAIN_LABELS, allow_pickle=True)
    x_val = np.load(FILE_VALIDATION, allow_pickle=True)
    validatelabels = np.load(FILE_VALIDATION_LABELS, allow_pickle=True)
    x_test = np.load(FILE_TEST, allow_pickle=True)
    testlabels = np.load(FILE_TEST_LABELS, allow_pickle=True)

# %% codecell

train_tuple = (x_train, trainlabels)
validate_tuple = (x_val, validatelabels)

dataset = (train_tuple, validate_tuple)
# Get the data.
(x_train, train_labels), (x_val, val_labels) = dataset

# %% [markdown]
#
# Here we define variables that will be used in the rest of this notebook
#


# %% codecell
learning_rate = 1e-4
epochs = 300
batch_size = 512
layers = 2
units = 16
dropout_rate = 0.5


# %% [markdown]
#
# To be able to create an efficient model, we need to know what kind of data we are working on.
# Thus the next section calculates some useful metrics about the data.
#


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

not_asshole = np.count_nonzero(train_labels == 0)
asshole = np.count_nonzero(train_labels)
print("There is ", asshole, " asshole and ", not_asshole, " not asshole")
total = not_asshole + asshole

not_asshole_val = np.count_nonzero(val_labels == 0)
asshole_val = np.count_nonzero(val_labels)
print("Validation: There is ", asshole_val, " asshole and ", not_asshole_val, " not asshole")


# %% [markdown]
#
# Finally the meat of the project.
# Here we start by studying the initial bias of the model.
# We compile a model and evaluate the result without training.

# %% codecell
# generetic model
model_no_init = mlp_model_generic.mlp_model(layers=layers,
                                            units=units,
                                            dropout_rate=dropout_rate,
                                            input_shape=x_train.shape[1:],
                                            num_classes=num_classes,
                                            learning_rate=learning_rate)

model_no_init.summary()


# %% codecell

model_no_init.predict(x_train[:10])

# %% markdown
# We can see that the loss is not great.

# %% codecell
# Init the bias

results_no_init = model_no_init.evaluate(x_train, train_labels, batch_size=batch_size, verbose=0)
print("Loss: {:0.4f}".format(results_no_init[0]))

# %% [markdown]
#
# Thus we calculate the inital bias and use it to improve our model.

# %% codecell

initial_bias = np.log([asshole/not_asshole])
initial_bias

# %% codecell

model = mlp_model_generic.mlp_model(layers=layers,
                                    units=units,
                                    dropout_rate=dropout_rate,
                                    input_shape=x_train.shape[1:],
                                    num_classes=num_classes,
                                    learning_rate=learning_rate,
                                    output_bias=initial_bias)

model.summary()


# %% [markdown]
# And we can see that the loss has been reduced.

# %% codecell
# Init the bias
model.predict(x_train[:10])


results = model.evaluate(x_train, train_labels, batch_size=batch_size, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

# %% [markdown]
# Let's save those inital weights to reuse them in the rest of this notebook.

# %% codecell
## Checkpoint the inital weights

initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)

# %% [markdown]
# Let's now make sure that the new weights help the learning by plotting the loss and other metrics
# First we create two models: one with and one without the weights.

# %% codecell
# Check that the ninit helped

model_no_init_test = mlp_model_generic.mlp_model(layers=layers,
                                                 units=units,
                                                 dropout_rate=dropout_rate,
                                                 input_shape=x_train.shape[1:],
                                                 num_classes=num_classes,
                                                 learning_rate=learning_rate)

model_no_init_test.load_weights(initial_weights)
model_no_init_test.layers[-1].bias.assign([0.0])
zero_bias_history = model_no_init_test.fit(
                                           x_train,
                                           train_labels,
                                           batch_size=batch_size,
                                           epochs=20,
                                           validation_data=(x_val, val_labels),
                                           verbose=1)

# %% codecell

model_init_test = mlp_model_generic.mlp_model(layers=layers,
                                              units=units,
                                              dropout_rate=dropout_rate,
                                              input_shape=x_train.shape[1:],
                                              num_classes=num_classes,
                                              learning_rate=learning_rate)

model_init_test.load_weights(initial_weights)
careful_bias_history = model_init_test.fit(
                                           x_train,
                                           train_labels,
                                           batch_size=batch_size,
                                           epochs=20,
                                           validation_data=(x_val, val_labels),
                                           verbose=1)

# %% [markdown]
# And we plot the metrics

# %% codecell
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history['loss'],
                 color=colors[n], label='Train '+label)
    plt.semilogy(history.epoch,  history.history['val_loss'],
                 color=colors[n], label='Val '+label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()


plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)
fig = plt.gcf()
fig.savefig('weight_init.jpg')

# %% [markdown]
# Now let's fully train the model on the maximum number of epochs and study the result.

# %% code cell
# let's train the model
# here we use the init weights

model = mlp_model_generic.mlp_model(layers=layers,
                                    units=units,
                                    dropout_rate=dropout_rate,
                                    input_shape=x_train.shape[1:],
                                    num_classes=num_classes,
                                    learning_rate=learning_rate)

# early_stopping = tf.keras.callbacks.EarlyStopping(
#                                                   monitor='val_loss',
#                                                   verbose=1,
#                                                   patience=2,
#                                                   mode='max',
#                                                   restore_best_weights=True)

# callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1)]

model.load_weights(initial_weights)
baseline_history = model.fit(
    x_train,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    # callbacks=callbacks,
    validation_data=(x_val, val_labels),
    verbose=1)


# %% codecell
def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        plt.ylim([0,1])

        plt.legend()


plot_metrics(baseline_history)


fig = plt.gcf()
fig.savefig('metrics.jpg')

# %% codecell
train_predictions_baseline = model.predict(x_train, batch_size=batch_size)
test_predictions_baseline = model.predict(x_test, batch_size=batch_size)


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


baseline_results = model.evaluate(x_test, testlabels,
                                  batch_size=batch_size, verbose=1)
for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()


# && codecell
plot_cm(testlabels, test_predictions_baseline)
fig = plt.gcf()
fig.savefig('cm.jpg')


# %% codecell
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')


plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", testlabels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')


# %% [markdown]
# # Class weight
#
# Here we use the class weight during fit

# %% codecell
# Label weights

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


class_weights = get_class_weight(train_labels)

# %% codecell
# train with class weight
model_weights = mlp_model_generic.mlp_model(layers=layers,
                                            units=units,
                                            dropout_rate=dropout_rate,
                                            input_shape=x_train.shape[1:],
                                            num_classes=num_classes,
                                            learning_rate=learning_rate)

# early_stopping = tf.keras.callbacks.EarlyStopping(
#                                                   monitor='val_loss',
#                                                   verbose=1,
#                                                   patience=2,
#                                                   mode='max',
#                                                   restore_best_weights=True)

# callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, verbose=1)]

model_weights.load_weights(initial_weights)
model_weights_history = model_weights.fit(
    x_train,
    train_labels,
    batch_size=batch_size,
    epochs=epochs,
    # callbacks=callbacks,
    validation_data=(x_val, val_labels),
    verbose=1,
    class_weight=class_weights
    )


# %% cellcode

plot_metrics(model_weights_history)

fig = plt.gcf()
fig.savefig('metrics_weighted.jpg')

train_predictions_weighted = model_weights.predict(x_train, batch_size=batch_size)
test_predictions_weighted = model_weights.predict(x_test, batch_size=batch_size)


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))


results_weighted = model_weights.evaluate(x_test, testlabels,
                                          batch_size=batch_size, verbose=1)
for name, value in zip(model.metrics_names, results_weighted):
    print(name, ': ', value)
print()


# && codecell
plot_cm(testlabels, test_predictions_weighted)
fig = plt.gcf()
fig.savefig('cm_weighted.jpg')


# %% codecell

plot_roc("Train Baseline", train_labels, train_predictions_weighted, color=colors[0])
plot_roc("Test Baseline", testlabels, test_predictions_weighted, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", testlabels, test_predictions_weighted, color=colors[1], linestyle='--')


plt.legend(loc='lower right')



# %% [markdown]
# The class weight seemed to improve the model but not enough that the results are interesting.
# Let's try oversampling the asshole dataset instead

# %% codecell

bool_arr_label = np.array(train_labels, dtype=bool)
asshole_features = x_train[bool_arr_label]
not_asshole_features = x_train[~bool_arr_label]

print(x_train.shape)
print(asshole_features.shape)
print(not_asshole_features.shape)

asshole_labels = train_labels[bool_arr_label]
not_asshole_labels = train_labels[~bool_arr_label]

ids = np.arange(len(asshole_features))
choices = np.random.choice(ids, len(not_asshole_features))

res_asshole_features = asshole_features[choices]
res_asshole_labels = asshole_labels[choices]

print(res_asshole_features.shape)
print(res_asshole_labels.shape)
print(not_asshole_features.shape)
print(not_asshole_labels.shape)

resampled_features = np.concatenate([res_asshole_features, not_asshole_features], axis=0)
resampled_labels = np.concatenate([res_asshole_labels, not_asshole_labels], axis=0)

print(resampled_features.shape)
print(resampled_labels.shape)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

print(resampled_features.shape)
print(resampled_labels.shape)
# print(res_pos_features.shape)
# print(neg_features.shape)

# %% codecell

resampled_model = mlp_model_generic.mlp_model(layers=layers,
                                              units=units,
                                              dropout_rate=dropout_rate,
                                              input_shape=x_train.shape[1:],
                                              num_classes=num_classes,
                                              learning_rate=learning_rate)
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1]
output_layer.bias.assign([0])

val_ds = tf.data.Dataset.from_tensor_slices((x_val, val_labels)).cache()
val_ds = val_ds.batch(batch_size).prefetch(2)

# resampled_steps_per_epoch = np.ceil(2.0*neg/batch_size)
# resampled_steps_per_epoch

resampled_history = resampled_model.fit(
    x=resampled_features,
    y=resampled_labels,
    epochs=epochs,
    # steps_per_epoch=resampled_steps_per_epoch,
    # callbacks = [early_stopping],
    validation_data=val_ds,
    verbose=1)


# %% codecell

plot_metrics(resampled_history)

fig = plt.gcf()
fig.savefig('metrics_resampled.jpg')

train_predictions_resampled = resampled_model.predict(x_train, batch_size=batch_size)
test_predictions_resampled = resampled_model.predict(x_test, batch_size=batch_size)


def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Actual not assholes (True Negatives): ', cm[0][0])
    print('Wrong judgement as asshole (False Positives): ', cm[0][1])
    print('Asshole missed (False Negatives): ', cm[1][0])
    print('True asshole (True Positives): ', cm[1][1])
    print('Total Assholes: ', np.sum(cm[1]))


results_resampled = resampled_model.evaluate(x_test, testlabels,
                                          batch_size=batch_size, verbose=1)
for name, value in zip(model.metrics_names, results_resampled):
    print(name, ': ', value)
print()


# && codecell
plot_cm(testlabels, test_predictions_resampled)
fig = plt.gcf()
fig.savefig('cm_resampled.jpg')


# %% codecell
# With early StopIteration

learning_rate = 1e-3
epochs = 300
batch_size = 512
layers = 3
units = 8
dropout_rate = 0.3


def mlp_model_test(layers, units, dropout_rate, input_shape, num_classes, learning_rate, output_bias=None):
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
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(8, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    model.add(tf.keras.layers.Dense(4, activation='relu'))

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


resampled_model_early = mlp_model_test(layers=layers,
                                              units=units,
                                              dropout_rate=dropout_rate,
                                              input_shape=x_train.shape[1:],
                                              num_classes=num_classes,
                                              learning_rate=learning_rate)
# resampled_model_early.load_weights(initial_weights)

resampled_model_early.summary()

# Reset the bias to zero, since this dataset is balanced.
output_layer_early = resampled_model_early.layers[-1]
output_layer_early.bias.assign([0])

# resampled_steps_per_epoch = np.ceil(2.0*neg/batch_size)
# resampled_steps_per_epoch

early_stopping = tf.keras.callbacks.EarlyStopping(
                                                  monitor='val_auc',
                                                  verbose=1,
                                                  patience=2,
                                                  mode='max',
                                                  restore_best_weights=True)

resampled_history_early = resampled_model_early.fit(
    x=resampled_features,
    y=resampled_labels,
    epochs=epochs,
    # steps_per_epoch=resampled_steps_per_epoch,
    callbacks=[early_stopping],
    validation_data=val_ds,
    verbose=1)

# %% codecell

plot_metrics(resampled_history_early)

fig = plt.gcf()
fig.savefig('metrics_resampled_early_d03_u16_8_4_l3.jpg')

train_predictions_resampled_early = resampled_model_early.predict(x_train, batch_size=batch_size)
test_predictions_resampled_early = resampled_model_early.predict(x_test, batch_size=batch_size)



results_resampled_early = resampled_model_early.evaluate(x_test, testlabels,
                                                         batch_size=batch_size, verbose=1)
for name, value in zip(resampled_model_early.metrics_names, results_resampled_early):
    print(name, ': ', value)
print()


plot_cm(testlabels, test_predictions_resampled_early)
fig = plt.gcf()
fig.savefig('cm_resampled_early_d03_u16_8_4_l3.jpg')

# %% cellcode

print(x_test[:2].todense())
