""" This script launches a uvicorn server for deploying a trained model via a REST API.
"""
import logging
import pyllars.logging_utils as logging_utils
logger = logging.getLogger(__name__)

from typing import Dict, Mapping

import argparse

from pepenc_model.data.peptide_encoder_embedding_dataset import PeptideEncoderEmbeddingDataset
from pepenc_model.models.peptide_encoder_lstm_embedding_model import PeptideEncoderLSTMEmbeddingModel

import pyllars.utils

from fastapi import FastAPI
app = FastAPI()

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__)
    parser.add_argument('config', help="The path to the yaml configuration file for the model")
    
    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)
    return args

config = "/prj/peptide-encoder-model/conf/base/config.yaml"
config = pyllars.utils.load_config(config)
model = PeptideEncoderLSTMEmbeddingModel(config)


@app.get("/transform/{peptides}")
def embed_peptides(peptides:str) -> Dict:
    """ Embed the given peptides, given as a comma-separated string

    Parameters
    ----------
    peptides : str
        A comma-separated list of peptides to embed

    Returns
    -------
    embedded_peptides : typing.Dict[]
    """

    dataset = PeptideEncoderEmbeddingDataset.create_from_csv_peptide_list(
        csv_peptide_list=peptides,
        aa_encoding_map=model.aa_encoding_map,
        max_len=config.get('max_sequence_length')
    )

    transformed_data = model.transform(dataset, progress_bar=True)

    ret = {
        'peptide_sequences': transformed_data.peptide_sequences.tolist(),
        'embedded_sequences': transformed_data.embedded_sequences.tolist()
    }

    return ret


def main():
    args = parse_arguments()
    config = pyllars.utils.load_config(args.config)

#if __name__ == '__main__':
#    main()
