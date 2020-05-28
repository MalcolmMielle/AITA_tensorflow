import sys
import os
script_path = os.path.dirname(__file__)
myutils_path = os.path.join(script_path, '..')
sys.path.append(myutils_path)

import mlp_model_generic
from utils import get_data
import numpy as np
import os


def main():
    script_path = os.path.dirname(__file__)
    lightweight = True
    if not lightweight:
        # PICKLE = "./google_tutorial_bag_of_word.pkl"
        DATA_PATH = os.path.join(script_path, "../../data/aita_clean.csv")
        FILE_TRAIN = "saved_data/train_google.npy"
        FILE_TRAIN_LABELS = "saved_data/train_google_labels.npy"
        FILE_VALIDATION = "saved_data/validation_google.npy"
        FILE_VALIDATION_LABELS = "saved_data/validation_google_labels.npy"
        FILE_TEST = "saved_data/test_google.npy"
        FILE_TEST_LABELS = "saved_data/test_google_labels.npy"
    else:
        # PICKLE = "./google_tutorial_bag_of_word.pkl"
        DATA_PATH = os.path.join(script_path, "../../data/aita_clean_light.csv")
        FILE_TRAIN = "saved_data/train_google_light.npy"
        FILE_TRAIN_LABELS = "saved_data/train_google_labels_light.npy"
        FILE_VALIDATION = "saved_data/validation_google_light.npy"
        FILE_VALIDATION_LABELS = "saved_data/validation_google_labels_light.npy"
        FILE_TEST = "saved_data/test_google_light.npy"
        FILE_TEST_LABELS = "saved_data/test_google_labels_light.npy"

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

        x_train, x_val, _ = mlp_model_generic.ngram_vectorize(traindataset, trainlabels, validatedataset)
        x_train = x_train.todense()
        x_val = x_val.todense()

        print(x_train.shape)
        print(x_val.shape)

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
        np.save(FILE_TEST, testdataset)
        np.save(FILE_TEST_LABELS, testlabels)

    else:
        print("LOAD PAST DATA")
        x_train = np.load(FILE_TRAIN, allow_pickle=True)
        trainlabels = np.load(FILE_TRAIN_LABELS, allow_pickle=True)
        x_val = np.load(FILE_VALIDATION, allow_pickle=True)
        validatelabels = np.load(FILE_VALIDATION_LABELS, allow_pickle=True)
        testdataset = np.load(FILE_TEST, allow_pickle=True)
        testlabels = np.load(FILE_TEST_LABELS, allow_pickle=True)

    #     print(x_train.shape)
    # x_train.todense()

    # %% codecell

    train_tuple = (x_train, trainlabels)
    validate_tuple = (x_val, validatelabels)

    dataset = (train_tuple, validate_tuple)

    # %% codecell
    # Train
    # learning_rate = 1e-4
    # epochs = 1000
    # batch_size = 128
    # layers = 2
    # units = 32
    # dropout_rate = 0.5

    # %% codecell
    learning_rate = 1e-4
    epochs = 1000
    batch_size = 128
    layers = 2
    units = 16
    dropout_rate = 0.5

    history = mlp_model_generic.train_ngram_model(dataset,
                                                  # num_train_labels=2,
                                                  # train_shape=x_train.shape[1:],
                                                  learning_rate=learning_rate,
                                                  epochs=epochs,
                                                  batch_size=batch_size,
                                                  layers=layers,
                                                  units=units,
                                                  dropout_rate=dropout_rate,
                                                  )

    # %% codecell

    import matplotlib.pyplot as plt
    # %matplotlib inline

    def plot_graphs(history, string):
        plt.plot(history[string])
        plt.plot(history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()

    plot_graphs(history, "acc")

    plot_graphs(history, "loss")


#  %% codecell
if __name__ == "__main__":
    main()
