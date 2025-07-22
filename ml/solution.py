from collab_v1.data_higgs import load_data, download_and_make_data
import pandas as pd
import tensorflow as tf
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Computer Modern Serif",
        "font.size": 24,
        "text.latex.preamble": "\n".join([r"\usepackage{siunitx}"]),
    }
)


def get_training_data(dataset):

    train = dataset["/train"]
    test = dataset["/valid"]

    X_train = train.drop(columns=["hlabel"])
    y_train = train["hlabel"]

    X_val = test.drop(columns=["hlabel"])
    y_val = test["hlabel"]

    return X_train, y_train, X_val, y_val


def catboost_model():
    return CatBoostClassifier(verbose=100, task_type="GPU")


def train_catboost(model, X_train, y_train, x_val, y_val):
    history = model.fit(
        X_train, y_train, eval_set=(x_val, y_val), early_stopping_rounds=50
    )

    return history


def overfitting_model(shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=shape,
            ),
            tf.keras.layers.Dense(
                256,
                use_bias=False,
            ),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            # tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, use_bias=False),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def simple_model(shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=shape,
            ),
            tf.keras.layers.Dense(50, use_bias=False),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(50, use_bias=False),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    return model


def normal_model(shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=shape,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                256,
                use_bias=False,
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["auc", "accuracy"]
    )

    return model


def multilayer_model(shape):
    model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=shape),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(16, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
    )   

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["auc", "accuracy"])

    return model



def train_dnn(
    model, X_train, y_train, x_val, y_val, epochs=20, batch_size=200, verbose=2
):
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=verbose,
    )

    return history


def draw_roc(y_pred, y_hat, ax, label):
    X, Y = [], []
    for i in range(0, 101, 10):
        thresh = i / 100
        y = y_pred > thresh
        mask_positive = y_hat.astype(bool)
        true_positive = (y[mask_positive]).sum()
        false_negative = (~y[mask_positive]).sum()

        mask_negative = ~(y_hat).astype(bool)
        false_positive = (y[mask_negative]).sum()
        true_negative = (~y[mask_negative]).sum()

        tpr = true_positive / (true_positive + false_negative)
        fpr = false_positive / (false_positive + true_negative)
        X.append(fpr)
        Y.append(tpr)

    ax.scatter(X, Y, label=label)


def roc_plots(*args):
    y_val = args[0]
    fig, ax = plt.subplots()

    for y, title in args[1:]:
        score = roc_auc_score(y_val, y)
        draw_roc(y, y_val, ax, title + f", auc={score:.3f}")

    x = np.linspace(0, 1, 20)
    ax.plot(x, x, "--", label="Random guessing")

    ax.legend(  )

    plt.grid()


def draw_loss(history, title=""):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    fig, ax = plt.subplots()
    ax.plot(loss, label="training loss")
    ax.plot(val_loss, label="validation loss")
    ax.set_title(title)
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend()


def create_histogram(data):
    fig, ax = plt.subplots(7, 4, figsize=(20, 20))
    for i, col in enumerate(data.columns):
        ax[i // 4, i % 4].hist(data[col], bins=100)
        ax[i // 4, i % 4].set_title(col)
    fig.tight_layout()


if __name__ == "__main__":
     # download_and_make_data()
    dataset = load_data()
    X_train, y_train, X_val, y_val = get_training_data(dataset)
    dataset.close()

    model = normal_model((X_train.shape[1],))
    train_dnn(model, X_train, y_train, X_val, y_val)

