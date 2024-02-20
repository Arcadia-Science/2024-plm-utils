import pathlib

import numpy as np
import sklearn
from Bio import SeqIO

RANDOM_STATE = 42


def load_and_filter_embeddings(embeddings_filepath, max_length=None):
    """
    Filter the given embeddings to remove any that correspond to sequences that are too long.
    """
    embeddings_filepath = pathlib.Path(embeddings_filepath)
    embeddings = np.load(embeddings_filepath)

    if max_length is None:
        return embeddings

    # TODO: a less hackish way to get the peptides fasta file corresponding to the embeddings.
    peptides_fasta_filepath = embeddings_filepath.with_name(
        embeddings_filepath.stem.replace("embeddings", "peptides.fa")
    )

    filtered_inds = [
        ind
        for ind, record in enumerate(SeqIO.parse(peptides_fasta_filepath, "fasta"))
        if len(record.seq) <= max_length
    ]
    return embeddings[filtered_inds, :]


def load_embeddings_and_labels(coding_filepath, noncoding_filepath, max_length=None):
    """
    Load embeddings from the given filepaths and create labels for coding/noncoding.
    """
    embs_coding = load_and_filter_embeddings(coding_filepath, max_length)
    embs_noncoding = load_and_filter_embeddings(noncoding_filepath, max_length)

    # create labels for the embeddings using 1 for 'coding' and 0 for 'noncoding'.
    labels_coding = np.ones(embs_coding.shape[0])
    labels_noncoding = np.zeros(embs_noncoding.shape[0])

    embs_all = np.concatenate([embs_coding, embs_noncoding], axis=0)
    labels_all = np.concatenate([labels_coding, labels_noncoding])

    return embs_all, labels_all


def train(
    coding_train_filepath,
    noncoding_train_filepath,
    coding_test_filepath=None,
    noncoding_test_filepath=None,
    max_length=None,
):
    """
    Train a random forest classifier using embeddings of peptides from coding and noncoding
    transcripts and print the validation and test metrics.

    Nomenclature note: we follow the sklearn convention of using `x` to denote
    the matrix of input features and `y` to denote the labels we are trying to predict.
    """

    x_all, y_all = load_embeddings_and_labels(
        coding_train_filepath, noncoding_train_filepath, max_length
    )

    # use PCA to reduce the dimensionality of the data to make training faster.
    # (`n_components` was chosen empirically from a plot of the explained variance.)
    pca = sklearn.decomposition.PCA(n_components=30)
    pca.fit(x_all)
    x_pcs = pca.transform(x_all)

    x_train, x_validation, y_train, y_validation = sklearn.model_selection.train_test_split(
        x_pcs, y_all, test_size=0.2, random_state=RANDOM_STATE
    )

    # `n_estimators` and `min_samples_split` were chosen empirically for fast training,
    # but performance doesn't seem to improve much with more estimators or deeper trees.
    # `class_weight="balanced"` is used to compensate for the class imbalance
    # between coding and noncoding transcripts.
    model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=30,
        min_samples_split=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    train_metrics = calc_metrics(y_train, y_train_pred)
    pretty_print_metrics(train_metrics, header="Training metrics")

    y_validation_pred = model.predict(x_validation)
    validation_metrics = calc_metrics(y_validation, y_validation_pred)
    pretty_print_metrics(validation_metrics, header="Validation metrics")

    if coding_test_filepath is not None and noncoding_test_filepath is not None:
        x_test, y_test = load_embeddings_and_labels(
            coding_test_filepath, noncoding_test_filepath, max_length
        )

        x_test_pcs = pca.transform(x_test)
        y_test_pred = model.predict(x_test_pcs)
        test_metrics = calc_metrics(y_test, y_test_pred)
        pretty_print_metrics(test_metrics, header="Test metrics")


def calc_metrics(y_true, y_pred):
    """
    Calculate performance metrics for the given true and predicted labels.
    """
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def pretty_print_metrics(metrics, header=None):
    """
    Print the given dict of metrics in a human-readable format.
    """
    output = "\n".join([f"{metric.capitalize()}: {value:.2f}" for metric, value in metrics.items()])

    if header is not None:
        output = f"{header}\n{output}"

    print(output)
