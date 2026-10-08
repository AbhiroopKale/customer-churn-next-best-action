"""A small Keras MLP wrapped as a scikit-learn classifier, so the ANN can be compared
with the other models using exactly the same Pipeline / cross-validation code."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasANN(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden=(32, 16), dropout=0.3, max_epochs=100, batch_size=64,
                 patience=8, random_state=42):
        self.hidden = hidden
        self.dropout = dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        layers = [tf.keras.layers.Input(shape=(X.shape[1],))]
        for units in self.hidden:
            layers += [tf.keras.layers.Dense(units, activation="relu"),
                       tf.keras.layers.Dropout(self.dropout)]
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.model_ = tf.keras.Sequential(layers)
        self.model_.compile(optimizer="adam", loss="binary_crossentropy")
        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                                restore_best_weights=True)
        # validation_split is taken from the data passed to fit() (i.e. training data only)
        self.model_.fit(X, y, validation_split=0.15, epochs=self.max_epochs,
                        batch_size=self.batch_size, callbacks=[stop], verbose=0)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = self.model_.predict(np.asarray(X, dtype="float32"), verbose=0).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
