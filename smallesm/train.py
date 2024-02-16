import numpy as np
import sklearn

RANDOM_STATE = 42


def load_data_and_labels(coding_filepath, noncoding_filepath):
    """
    Load embeddings from the given filepaths and create labels for the data.
    """
    x_coding = np.load(coding_filepath)
    x_noncoding = np.load(noncoding_filepath)

    # create labels for the data using 1 for coding and 0 for noncoding sequences.
    labels_coding = np.ones(x_coding.shape[0])
    labels_noncoding = np.zeros(x_noncoding.shape[0])

    X = np.concatenate([x_coding, x_noncoding], axis=0)
    y = np.concatenate([labels_coding, labels_noncoding])

    return X, y


def train(
    coding_train_filepath,
    noncoding_train_filepath,
    coding_test_filepath=None,
    noncoding_test_filepath=None,
):
    X, y = load_data_and_labels(coding_train_filepath, noncoding_train_filepath)

    # use PCA to reduce the dimensionality of the data to make training faster.
    # (`n_components` was chosen empirically from a plot of the explained variance.)
    pca = sklearn.decomposition.PCA(n_components=30)
    pca.fit(X)

    X_pcs = pca.transform(X)

    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        X_pcs, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Train a random forest classifier using params optimized for speed.
    model = sklearn.ensemble.RandomForestClassifier(
        n_estimators=30,
        min_samples_split=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    valid_score = model.score(X_valid, y_valid)

    print(f"Training accuracy: {train_score:.2f}")
    print(f"Validation accuracy: {valid_score:.2f}")

    # If test data is provided, evaluate the model on the test set.
    if coding_test_filepath is not None and noncoding_test_filepath is not None:
        X_test, y_test = load_data_and_labels(coding_test_filepath, noncoding_test_filepath)

        X_test_pcs = pca.transform(X_test)
        test_score = model.score(X_test_pcs, y_test)
        print(f"Test accuracy: {test_score:.2f}")
