""" This module contains a pytorch dataset for extracting peptide embeddings.

In particular, each "instance" of the dataset comprises a trimmed peptide sequence, both the amino acid sequence as well
as the encoded version for input to the network.
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np

import torch
import torch.utils.data

from lifesci.peptide_dataset import PeptideDataset
import pyllars.string_utils as string_utils

from typing import NamedTuple

class PeptideEncoderEmbeddingDatasetItem(NamedTuple):
    peptide_sequence: str
    encoded_sequence: torch.IntTensor

_DEFAULT_SEQUENCE_COLUMN = 'sequence'
_DEFAULT_NAME = "PeptideEncoderEmbeddingDataset"
_DEFAULT_MAX_LEN = 25

class PeptideEncoderEmbeddingDataset(torch.utils.data.Dataset):
    """ Generate input samples from a list of amino acid sequences

    Parameters
    ----------
    dataset_path : str
        The path to the dataset. It should be compatible with `pandas.read_csv`
        and contain a column named `sequence_column` which includes the
        sequences. Other columns are ignored.

    aa_encoding_map : pyllars.string_utils.encoding_map_type
        A mapping from each amino acid to its integer index.

        N.B. This should **not** be a one-hot representation, but, as stated,
        the integer index. Further, the padding character must be "-".

    sequence_column : str
        The name of the column which contains the amino acid sequences

    max_len : int
        The maximum length for a peptide. Peptides longer than this will be
        truncated, and shorter peptides will be padded to this length.

    name : str
        A name for the dataset instance. This is mostly used for logging.
    """    
    def __init__(self,
            dataset_path:str,
            aa_encoding_map:string_utils.encoding_map_type,
            sequence_column:str=_DEFAULT_SEQUENCE_COLUMN,
            max_len:int=_DEFAULT_MAX_LEN,
            name:str=_DEFAULT_NAME):

        self.aa_encoding_map = aa_encoding_map
        self.sequence_column = sequence_column
        self.max_len = max_len
        self.name = name

        df_peptides = PeptideDataset.load(dataset_path, sequence_column, filters=["standard_aa_only"])
        self.aa_sequences = df_peptides[self.sequence_column].values

        self.encoded_aa_sequences = string_utils.encode_all_sequences(
            sequences=self.aa_sequences,
            encoding_map=self.aa_encoding_map,
            maxlen=self.max_len,
            pad_value='-',
            same_length=False
        )
        self.encoded_aa_sequences = self.encoded_aa_sequences.astype(int)

    def log(self, msg:str, level:int=logging.INFO) -> None:
        """ Log `msg` using `level` using the module-level logger """    
        msg = "[{}] {}".format(self.name, msg)
        logger.log(level, msg)

    def __len__(self) -> int:
        return len(self.aa_sequences)

    def __getitem__(self, idx) -> PeptideEncoderEmbeddingDatasetItem:
        peptide_seq = self.aa_sequences[idx]
        encoded_seq = self.encoded_aa_sequences[idx]
        encoded_seq = torch.as_tensor(encoded_seq, dtype=torch.long)

        ret = PeptideEncoderEmbeddingDatasetItem(
            peptide_seq, encoded_seq
        )

        return ret

    def get_trimmed_peptide_lengths(self, peptides) -> np.ndarray:
        """ Extract the trimmed length of the given peptides, which accounts for max_len """
        peptide_lengths = [len(p) for p in peptides]
        trimmed_peptide_lengths = np.clip(peptide_lengths, 0, self.max_len)
        return trimmed_peptide_lengths