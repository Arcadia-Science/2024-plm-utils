import pathlib

import pytest


@pytest.fixture
def artifacts_dirpath():
    return pathlib.Path(__file__).parent / "data"
