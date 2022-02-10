# Peptide Encoder Model

An encoder for peptides (short amino acid sequences) trained using [`pepenc`](https://github.com/bmmalone/peptide-encoder)

This package wraps the single best model learned using hyperparameter optimization and exposes it via a Docker image.

### Installation

This project is written in `python3` and can be installed with `pip`.

```
pip3 install .
```

**Prerequisites**: This project relies on quite a few prerequisites, such as [pepenc](https://github.com/bmmalone/peptide-encoder),
pytorch, cudnn, pyarrow, and others. Further, this is mostly a hobby project. Consequently, the `requirements.txt` file
does not include a complete set of prerequisites at this time. Ideally, this is updated in the future....

### Usage

After installation, a trained model (from `pepenc`) can be used to embed peptides using a command similar to the
following.

```
embed-peptides /prj/peptide-encoder-model/conf/base/config.yaml /prj/peptide-encoder-model/data/raw/sample-peptides.test.csv /prj/peptide-encoder-model/data/processed/embeddings.parquet --logging-level INFO
```

The `--help` flag can be used to see a description of all arguments to the script.

For adjusting the hyperparameter search space, algorithms, or schedulers, the `pepenc/models/train_pepenc_models.py`
script can be adjusted. If the package was not installed in `pip` "editable" mode, then make sure to re-run `pip install`
so that the changes take effect for the next run of ray.

### Documentation

Unfortunately, see "Prerequisites", above. The file `conf/base/config.yaml` shows examples of all hyperparameters, data
files, etc., for training models.

#### Input data format

The models in this project require an input csv file that has one row which is a header and remaining rows which are
the peptides for the various datasets. The column in the csv file with the peptide sequences must be named `sequence`.
(This can be adjusted if calling the `pepenc` library from python code.)

#### Output data format

The embedded peptide sequences are saved as a data frame in parquet format. One column is named `peptide_sequence`, and
it contains the original amino acid sequence as a string. A second column named `embedded_sequence` contains the
embedding for the associated peptide as a 1-dimensional numpy array.