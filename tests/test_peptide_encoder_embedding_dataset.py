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

import torch.utils.data

import pepenc_model
import pepenc_model.data.data_utils as data_utils
from pepenc_model.data.peptide_encoder_embedding_dataset import PeptideEncoderEmbeddingDataset

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

def test_dataset_loading_from_file(config:Mapping) -> None:
    """ Test loading a peptide dataset from a file """

    dataset_path = config.get('test_set')
    encoding_map = data_utils.load_encoding_map()
    dataset = PeptideEncoderEmbeddingDataset(dataset_path, encoding_map)

    expected_length = 1981 # known from file
    assert len(dataset) == expected_length

    batch_size = config.get('batch_size')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    it = iter(data_loader)
    data = next(it)

    assert len(data.peptide_sequence) == batch_size
    assert data.encoded_sequence.shape[0] == batch_size


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
    test_dataset_loading_from_file(config)

if __name__ == '__main__':
    run_all()