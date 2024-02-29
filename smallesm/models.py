import pathlib

import joblib
import numpy as np
import sklearn
import sklearn.decomposition
import sklearn.ensemble
import sklearn.model_selection

RANDOM_STATE = 42


def calc_metrics(y_true, y_pred_proba):
    """
    Calculate performance metrics for the given true and predicted labels.

    y_true : array-like of shape (n_samples,)
        The true binary labels.
    y_pred_proba : array-like of shape (n_samples, 2)
        The predicted probabilities for the negative and positive classes
        (this is the output by the `predict_proba` method of sklearn classifiers).
    """
    # the probability of the positive class is in the second column.
    y_pred_proba = y_pred_proba[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)

    # `roc_auc_score` raises a ValueError if only one class is present in `y_true`.
    try:
        auc_roc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc_roc = np.nan

    return {
        "auc_roc": auc_roc,
        "accuracy": sklearn.metrics.accuracy_score(y_true, y_pred),
        "precision": sklearn.metrics.precision_score(y_true, y_pred),
        "recall": sklearn.metrics.recall_score(y_true, y_pred),
        "mcc": sklearn.metrics.matthews_corrcoef(y_true, y_pred),
        "num_true_positive": ((y_true == 1) & (y_pred == 1)).sum(),
        "num_false_positive": ((y_true == 0) & (y_pred == 1)).sum(),
        "num_true_negative": ((y_true == 0) & (y_pred == 0)).sum(),
        "num_false_negative": ((y_true == 1) & (y_pred == 0)).sum(),
        "num_positive": (y_true == 1).sum(),
        "num_negative": (y_true == 0).sum(),
    }


def pretty_print_metrics(metrics, header=None):
    """
    Print the given dict of metrics in a human-readable format.
    """
    output = "\n".join([f"{metric.capitalize()}: {value:.2f}" for metric, value in metrics.items()])

    if header is not None:
        output = f"{header}\n{output}"

    print(output)


class EmbeddingsClassifier:
    """
    A minimal approach to defining and training a binary classifier given a matrix of embeddings.

    It first reduces the dimensionality of the embeddings using PCA,
    then trains a random forest classifier on the reduced embeddings.

    This approach was designed for predicting whether transcripts are coding or noncoding
    from the ESM embeddings of the longest putative ORF from each transcript,
    but it is general enough to be used with any kind of feature matrix.
    """

    def __init__(self, pca, classifier, verbose=False):
        self.pca = pca
        self.classifier = classifier
        self.verbose = verbose

    def save(self, model_dirpath):
        """
        Save the model to the given directory.
        """
        model_dirpath = pathlib.Path(model_dirpath)
        model_dirpath.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.pca, model_dirpath / "pca.joblib")
        joblib.dump(self.classifier, model_dirpath / "classifier.joblib")

    @classmethod
    def load(cls, model_dirpath, **kwargs):
        """
        Load a pre-trained model from the given directory.
        """
        model_dirpath = pathlib.Path(model_dirpath)
        pca = joblib.load(model_dirpath / "pca.joblib")
        classifier = joblib.load(model_dirpath / "classifier.joblib")
        return cls(pca, classifier, **kwargs)

    @classmethod
    def init(cls, n_components=30, n_estimators=30, min_samples_split=10, **kwargs):
        """
        Initialize a new, untrained model with the given hyperparameters.
        A 'model' consists of PCA to reduce dimensionality followed by a random forest classifier.

        This method is separate from `__init__` because `__init__` is also used
        when loading a pre-trained model (whose hyperparameters are already set).

        Parameters
        ----------
        n_components : int
            The number of principal components to retain after PCA.
        n_estimators : int
            The number of trees in the random forest.
        min_samples_split : int
            The minimum number of samples required to split an internal node
            (this indirectly controls the depth of the trees).

        Notes
        -----
        - the default value of `n_components` was chosen empirically using a test dataset
            of embeddings from the smallest (8m) ESM-2 model.
        - the default values of `n_estimators` and `min_samples_split` were chosen empirically
            for fast training, but (anecdotally) model performance doesn't improve much
            with either more estimators or deeper trees.
        """
        pca = sklearn.decomposition.PCA(n_components=n_components)

        # `class_weight="balanced"` is used to compensate for class imbalance in the training data.
        classifier = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        return cls(pca, classifier, **kwargs)

    def train(self, x, y, test_size=0.2, random_state=RANDOM_STATE):
        """
        x: the matrix of embeddings
        y: the corresponding labels (1 for the positive class, 0 for the negative class)

        Nomenclature note: we follow the sklearn convention of using `x` to denote
        the matrix of input features and `y` to denote the labels we are trying to predict.
        """
        self.pca.fit(x)
        x_pcs = self.pca.transform(x)

        x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(
            x_pcs, y, test_size=test_size, random_state=random_state
        )

        self.classifier.fit(x_train, y_train)

        y_validation_pred = self.classifier.predict_proba(x_validation)
        validation_metrics = calc_metrics(y_validation, y_validation_pred)
        if self.verbose:
            pretty_print_metrics(validation_metrics, header="Validation metrics")

        return validation_metrics

    def predict(self, x):
        """
        Return binary predictions for the given matrix of embeddings.
        (1 for the positive class, 0 for the negative class)
        """
        x_pcs = self.pca.transform(x)
        return self.classifier.predict(x_pcs)

    def predict_proba(self, x):
        """
        Return prediction probabilities for the given matrix of embeddings.
        """
        x_pcs = self.pca.transform(x)
        return self.classifier.predict_proba(x_pcs)
