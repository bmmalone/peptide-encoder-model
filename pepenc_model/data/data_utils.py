""" This module contains helpers for automatically loading data files stored in the pepenc_model/data folder. It is
mostly intended for use in testing.
"""
import joblib
import os
import pathlib
import yaml

from lifesci.peptide_dataset import PeptideDataset

from typing import Mapping

def _get_base_data_dir() -> pathlib.Path:
    import pepenc_model.data # that's me!
    base_data_dir = os.path.dirname(pepenc_model.data.__file__)
    base_data_dir = pathlib.Path(base_data_dir)
    #base_data_dir = base_data_dir.parent
    return base_data_dir

###
# Paths to files in the `pepenc_model/data` directory
###

def get_sample_config_path():
    path = _get_base_data_dir() / "conf" / "base" / "config.yaml"
    return str(path)

def get_encoding_map_path():
    path = _get_base_data_dir() / "intermediate" / "oh-aa-encoding-map.jpkl"
    return str(path)

def get_sample_test_peptides_path():
    path = _get_base_data_dir() / "raw" / "sample-peptides.test.csv"
    return str(path)

def get_network_dir():
    path = _get_base_data_dir() / "models" / "ray_checkpoint"
    return str(path)

###
# Helpers for loading the distributed data files
###
def load_encoding_map() -> Mapping:
    p = get_encoding_map_path()
    encoding_map = joblib.load(p)
    return encoding_map

def load_peptides_as_csv_string() -> str:
    """ Load the peptides in the test data file and return as a csv string """
    sequence_column = "sequence"

    f = get_sample_test_peptides_path()
    df_peptides = PeptideDataset.load(f, sequence_column, filters=["standard_aa_only"])
    aa_sequences = df_peptides[sequence_column].values
    csv_peptide_list = ",".join(aa_sequences)
    return csv_peptide_list


def load_sample_config(update_paths:bool=True) -> Mapping:
    """ Load the config file and, optionally, update the path values"""
    path = get_sample_config_path()

    with open(path) as f:
        config = yaml.full_load(f)

    if update_paths:
        config['aa_encoding_map'] = get_encoding_map_path()
        config['test_set'] = get_sample_test_peptides_path()
        config['network_dir'] = get_network_dir()

    return config