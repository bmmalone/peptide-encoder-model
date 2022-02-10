""" This script uses a trained peptide encoder LSTM model to embed peptides. The output is a parquet file with one
column named `peptide_sequence`, which contains the original amino acid sequence, and a second column named
`embedded_sequence`, which contains the embedding for the associated peptide.
"""
import logging
import pyllars.logging_utils as logging_utils
logger = logging.getLogger(__name__)

from typing import Mapping

import argparse
import pandas as pd
import pyllars.shell_utils as shell_utils
import pyllars.utils

from pepenc_model.data.peptide_encoder_embedding_dataset import PeptideEncoderEmbeddingDataset
from pepenc_model.models.peptide_encoder_lstm_embedding_model import PeptideEncoderLSTMEmbeddingModel


def embed_dataset(config:Mapping, dataset_path:str) -> pd.DataFrame:
    """ Load the model and use it to embed the peptides in the dataset
    
    Parameters
    ----------
    config : typing.Mapping
        The configuration for the trained model

    dataset_path : str
        The complete path to the dataset file

    Returns
    -------
    df_embedded : pandas.DataFrame
        A data frame containing the peptide sequences and the associated embeddings
    """
    msg = "Loading the trained model"
    logger.info(msg)
    model = PeptideEncoderLSTMEmbeddingModel(config)

    msg = "Loading the dataset"
    logger.info(msg)
    dataset = PeptideEncoderEmbeddingDataset(dataset_path, model.aa_encoding_map)

    msg = "Embedding the peptides"
    logger.info(msg)
    transformed_data = model.transform(dataset, progress_bar=True)

    msg = "Creating the data frame"
    logger.info(msg)

    df_embedded = pd.DataFrame()
    df_embedded['peptide_sequence'] = transformed_data.peptide_sequences
    df_embedded['embedded_sequence'] = transformed_data.embedded_sequences.tolist()

    return df_embedded

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('config', help="The path to the yaml configuration file for the model")
    parser.add_argument('dataset', help="The path to the peptide dataset")
    parser.add_argument('out', help="The path to the output parquet file")

    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)
    return args

def main():
    args = parse_arguments()
    config = pyllars.utils.load_config(args.config)
    
    df_embedded = embed_dataset(config, args.dataset)

    msg = f"Writing embeddings to disk: '{args.out}'"
    logger.info(msg)
    shell_utils.ensure_path_to_file_exists(args.out)
    df_embedded.to_parquet(args.out)

if __name__ == '__main__':
    main()
