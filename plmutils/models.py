import pathlib

import joblib
import sklearn
import sklearn.decomposition
import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline

RANDOM_STATE = 42


def calc_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate performance metrics for the given true and predicted labels.

    y_true : array-like of shape (n_samples,)
        The true binary labels.
    y_pred_proba : array-like of shape (n_samples,)
        The predicted probabilities for the positive classes
        (this is the second column of the array returned by the `predict_proba` method
        of sklearn classifiers).
    """
    y_pred = (y_pred_proba > threshold).astype(int)

    return {
        "auc_roc": sklearn.metrics.roc_auc_score(y_true, y_pred_proba),
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
    then trains a random forest classifier on the matrix of principal components.
    """

    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose

    def save(self, model_dirpath):
        """
        Save the model to the given directory.
        """
        model_dirpath = pathlib.Path(model_dirpath)
        model_dirpath.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.model, model_dirpath / "model.joblib")

    @classmethod
    def load(cls, model_dirpath, **kwargs):
        """
        Load a pre-trained model from the given directory.
        """
        model_dirpath = pathlib.Path(model_dirpath)
        model = joblib.load(model_dirpath / "model.joblib")
        return cls(model, **kwargs)

    @classmethod
    def create(cls, n_components=30, n_estimators=30, min_samples_split=10, **kwargs):
        """
        Create a new, untrained model with the given hyperparameters.
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
        model = sklearn.pipeline.Pipeline(
            [
                ("pca", sklearn.decomposition.PCA(n_components=n_components)),
                (
                    "classifier",
                    sklearn.ensemble.RandomForestClassifier(
                        n_estimators=n_estimators,
                        min_samples_split=min_samples_split,
                        # Compensate for the class imbalance in the training data.
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        return cls(model, **kwargs)

    def train(self, X, y, test_size=0.2, random_state=RANDOM_STATE):
        """
        X: the matrix of embeddings
        y: the corresponding labels (1 for the positive class, 0 for the negative class)

        Nomenclature note: we follow the sklearn convention of using `X` to denote
        the matrix of input features and `y` to denote the labels we are trying to predict.
        """
        X_train, X_validation, y_train, y_validation = sklearn.model_selection.train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)

        y_validation_pred = self.model.predict_proba(X_validation)
        validation_metrics = calc_metrics(y_validation, y_validation_pred[:, 1], threshold=0.5)
        if self.verbose:
            pretty_print_metrics(validation_metrics, header="Validation metrics")

        return validation_metrics
