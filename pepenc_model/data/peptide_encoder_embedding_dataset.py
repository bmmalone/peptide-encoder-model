""" This module contains a pytorch dataset for extracting peptide embeddings.

In particular, each "instance" of the dataset comprises a trimmed peptide sequence, both the amino acid sequence as well
as the encoded version for input to the network.
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
import torch.utils.data

import lifesci.peptide_dataset
from lifesci.peptide_dataset import PeptideDataset
import pyllars.string_utils as string_utils

from typing import NamedTuple, Sequence

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
    aa_sequences : typing.Sequence[str]
        The list of amino acid sequences

    aa_encoding_map : pyllars.string_utils.encoding_map_type
        A mapping from each amino acid to its integer index.

    max_len : int
        The maximum length for a peptide. Peptides longer than this will be
        truncated, and shorter peptides will be padded to this length.

    name : str
        A name for the dataset instance. This is mostly used for logging.
    """    
    def __init__(self,
            aa_sequences:Sequence[str],
            aa_encoding_map:string_utils.encoding_map_type,
            max_len:int=_DEFAULT_MAX_LEN,
            name:str=_DEFAULT_NAME):

        self.aa_sequences = aa_sequences
        self.max_len = max_len
        self.name = name

        encoded_aa_sequences = string_utils.encode_all_sequences(
            sequences=aa_sequences,
            encoding_map=aa_encoding_map,
            maxlen=max_len,
            pad_value='-',
            same_length=False
        )
        self.encoded_aa_sequences = encoded_aa_sequences.astype(int)

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

    @classmethod
    def create_from_file(clazz,
            dataset_path:str,
            aa_encoding_map:string_utils.encoding_map_type,
            sequence_column:str=_DEFAULT_SEQUENCE_COLUMN,
            max_len:int=_DEFAULT_MAX_LEN,
            name:str=_DEFAULT_NAME) -> "PeptideEncoderEmbeddingDataset":
        """ Load the peptide dataset from the given file

        All peptides with non-standard amino acids are automatically filtered.

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

        Returns
        -------
        dataset : pepenc_model.data.peptide_encoder_embedding_dataset.PeptideEncoderEmbeddingDataset
            The loaded dataset
        """

        df_peptides = PeptideDataset.load(dataset_path, sequence_column, filters=["standard_aa_only"])
        aa_sequences = df_peptides[sequence_column].values

        ds = clazz(aa_sequences, aa_encoding_map, max_len, name)
        return ds

    @classmethod
    def create_from_csv_peptide_list(clazz,
            csv_peptide_list:str,
            aa_encoding_map:string_utils.encoding_map_type,
            max_len:int=_DEFAULT_MAX_LEN,
            name:str=_DEFAULT_NAME) -> "PeptideEncoderEmbeddingDataset":
        """ Load the peptide dataset from the given file

        All peptides with non-standard amino acids are automatically filtered.

        Parameters
        ----------
        csv_peptide_list : str
            A list of peptides separated by commas

        aa_encoding_map : pyllars.string_utils.encoding_map_type
            A mapping from each amino acid to its integer index.

            N.B. This should **not** be a one-hot representation, but, as stated,
            the integer index. Further, the padding character must be "-".

        max_len : int
            The maximum length for a peptide. Peptides longer than this will be
            truncated, and shorter peptides will be padded to this length.

        name : str
            A name for the dataset instance. This is mostly used for logging.

        Returns
        -------
        dataset : pepenc_model.data.peptide_encoder_embedding_dataset.PeptideEncoderEmbeddingDataset
            The loaded dataset
        """
        aa_sequences = csv_peptide_list.split(",")
        aa_sequences = pd.Series(aa_sequences)
        m_valid_aa = lifesci.peptide_dataset._standard_aa_only_filter(aa_sequences)
        aa_sequences = aa_sequences[m_valid_aa]
        ds = clazz(aa_sequences, aa_encoding_map, max_len, name)
        return ds

