""" This module contains a wrapper class for the PeptideEncoderLSTM network. It is specifically for answering
transformation queries. That is, this class assumes the network has already been trained (presumably with the `pepenc`
project).
"""
import logging
logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pathlib
import torch
import tqdm

import pyllars.utils
import pyllars.torch.torch_utils as torch_utils

import pyllars.validation_utils as validation_utils

validation_utils.validate_type

from pepenc.models.peptide_encoder_lstm_net import PeptideEncoderLSTMNetwork
from pepenc_model.data.peptide_encoder_embedding_dataset import (
    PeptideEncoderEmbeddingDataset, PeptideEncoderEmbeddingDatasetItem
)

from typing import Mapping, NamedTuple

_DEFAULT_BATCH_SIZE = 64
_DEFAULT_DEVICE_NAME = None
_DEFAULT_NAME = "PeptideEncoderLSTMEmbeddingModel"

class PeptideEncoderEmbeddingResults(NamedTuple):
    peptide_sequences: np.ndarray
    encoded_sequences: np.ndarray
    embedded_sequences: np.ndarray

class PeptideEncoderLSTMEmbeddingModel(object):
    """ The peptide encoder embedding model

    Parameters
    ----------
    config : typing.Mapping
        The configuration options. The following options are recognized.

        * `network_dir` : the path to the **folder** containing the trained model. Presumably, this is a checkpoint
            folder from Ray.

        * `aa_encoding_map`: the path to a joblib'd dictionary containing the index encoding for each amino acid

        * `device_name` : [Optional] the name of the device for pytorch. The code typically detects if a GPU is
            available and uses it when possible.

        * `batch_size` :  [Optional] the batch size for running the network

        * `name` : [Optional] a name for this model. This is mostly used for logging
    """
    def __init__(self, config:Mapping) -> "PeptideEncoderLSTMEmbeddingModel":
        self.config = config
        self.name = config.get('name', _DEFAULT_NAME)
        self.batch_size = config.get('batch_size', _DEFAULT_BATCH_SIZE)
        self.device_name = config.get('device_name', _DEFAULT_DEVICE_NAME)

        self.device = torch_utils.get_device(self.device_name)

        self._validate_config()
        self._load_encoding_map()
        self._load_network()


    def log(self, msg:str, level:int=logging.INFO):    
        """ Log `msg` using `level` using the module-level logger """    
        msg = "[{}] {}".format(self.name, msg)
        logger.log(level, msg)

    def _validate_config(self):
        """ Ensure all of the required keys are present in the configuration """
        required_keys = ['aa_encoding_map', 'network_dir']
        validation_utils.check_keys_exist(self.config, required_keys, 'config', self.name)

    def _load_encoding_map(self):
        """ Load the encoding map for preparing amino acid sequences for the network """
        aa_encoding_map = self.config.get('aa_encoding_map')        
        self.aa_encoding_map = joblib.load(aa_encoding_map)

    def _load_network(self) -> None:
        """ Load the trained network """

        self.network_dir = self.config.get('network_dir')
        self.network_dir = pathlib.Path(self.network_dir)
        
        checkpoint_path = str(self.network_dir / 'checkpoint.pt')
        state_dict = torch.load(checkpoint_path)

        msg = "restoring the network"
        self.log(msg)

        network_config = self.network_dir / "params.json"
        network_config = pyllars.utils.load_config(network_config)
        self.network_config = network_config

        net = PeptideEncoderLSTMNetwork(network_config)
        net.load_state_dict(state_dict)

        self.net = torch_utils.send_network_to_device(net, self.device)

    def transform(self, dataset:PeptideEncoderEmbeddingDataset,
            progress_bar:bool=False) -> PeptideEncoderEmbeddingResults:
        """ Embed the peptides using the trained network """

        self.net.eval() # make sure the network knows we are making predictions

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        it = enumerate(data_loader)
        if progress_bar:
            it = tqdm.tqdm(it)

        # keep track of all predictions
        all_peptide_sequences = []
        all_encoded_sequences = []
        all_embedded_sequences = []

        for i, data in it:
            data_to_device = torch_utils.send_data_to_device(
                *data, device=self.device
            )
            data_to_device = PeptideEncoderEmbeddingDatasetItem(*data_to_device)

            lengths = dataset.get_trimmed_peptide_lengths(data_to_device.peptide_sequence)
            embedded_sequences = self.net(data_to_device.encoded_sequence, lengths)
            embedded_sequences = torch_utils.retrieve_data_from_device(embedded_sequences)
            embedded_sequences = embedded_sequences[0] # the helper always returns a list
            
            all_peptide_sequences.append(data.peptide_sequence)
            all_encoded_sequences.append(data.encoded_sequence)
            all_embedded_sequences.append(embedded_sequences)

        ret = PeptideEncoderEmbeddingResults(
            np.concatenate(all_peptide_sequences),
            np.concatenate(all_encoded_sequences),
            np.concatenate(all_embedded_sequences)
        )

        return ret
