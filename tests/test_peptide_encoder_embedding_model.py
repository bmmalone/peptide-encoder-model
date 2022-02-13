""" A high-level set of tests for checking the peptide encoder embedding dataset
"""

import pytest

import argparse
import logging
logger = logging.getLogger(__name__)

###
# If using the pyllars package, these lines can be used so that the logger
# actually outputs logging statements across all modules.
###
import pyllars.logging_utils as logging_utils
logging_utils.set_logging_values(logging_level='DEBUG')

import pandas as pd

import pepenc_model
import pepenc_model.data.data_utils as data_utils
from pepenc_model.data.peptide_encoder_embedding_dataset import PeptideEncoderEmbeddingDataset
from pepenc_model.models.peptide_encoder_lstm_embedding_model import PeptideEncoderLSTMEmbeddingModel

from typing import Mapping

###
# Fixtures and other "Arrange" phase helpers
###
def get_config() -> Mapping:
    config = data_utils.load_sample_config()
    return config

@pytest.fixture
def config() -> Mapping:
    return get_config()

def get_version() -> str:
    return pepenc_model.__version__

@pytest.fixture
def version() -> str:
    return get_version()

###
# The actual tests
###

def test_create_model(config:Mapping) -> None:
    """ Test creating a model from a configuration file """

    model = PeptideEncoderLSTMEmbeddingModel(config)
    assert model.batch_size == config.get('batch_size')

def test_model_transform_file_dataset(config:Mapping) -> None:
    """ Test transforming a dataset from a file with a loaded model """    
    model = PeptideEncoderLSTMEmbeddingModel(config)

    dataset_path = config.get('test_set')
    dataset = PeptideEncoderEmbeddingDataset.create_from_file(
        dataset_path=dataset_path,
        aa_encoding_map=model.aa_encoding_map,
        max_len=config.get('max_sequence_length')
    )

    transformed_data = model.transform(dataset, progress_bar=True)

    df_embedded = pd.DataFrame()
    df_embedded['peptide_sequence'] = transformed_data.peptide_sequences
    df_embedded['embedded_sequence'] = transformed_data.embedded_sequences.tolist()

    assert len(df_embedded) == len(dataset)
    

def test_version(version:str) -> None:
    """ Ensure we have the correct version

    N.B. Since we have a fixture called `version`, that will be passed as
    the "`version`" parameter when running through pytest.

    Parameters
    ----------
    version : str
        A string representation of the version of `aa_encode`

    Returns
    -------
    None : None
        We assert that the version matches our expected version
    """

    # we do not really have an "Act" phase in this simple test

    # "Assert" that we have the expected behavior
    expected_version = '0.1.1'
    assert (expected_version == version)
    
def run_all():
    """ Run all of the tests

    This function is useful in case we want to run our tests outside of the pytest framework.
    """
    # since we are not running through pytest, we have to grab the inputs to the tests
    version = get_version()
    test_version(version)

    config = get_config()
    test_create_model(config)
    test_model_transform_file_dataset(config)

if __name__ == '__main__':
    run_all()