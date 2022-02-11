# Peptide Encoder Model

An encoder for peptides (short amino acid sequences) trained using [`pepenc`](https://github.com/bmmalone/peptide-encoder)

This package wraps the single best model learned using hyperparameter optimization and exposes it via a Docker image.

### Installation

This project is written in `python3` and can be installed with `pip`.

```
git clone https://github.com/bmmalone/peptide-encoder-model.git
cd piptide-encoder-model
pip3 install -r requirements.txt .
```

(The "period" at the end is required.)

**Prerequisites**: This project relies on quite a few prerequisites, such as [pepenc](https://github.com/bmmalone/peptide-encoder),
pytorch, ray, cudnn, and others. Both the `requirements.txt` and `setup.py` files aim to install these dependencies
correctly; nevertheless, it may still be preferable to install these dependencies before installing this package.

In particular, the `find-links` argument to pip may need to be adjusted depending on the available version of CUDA.

### Usage

After installation, a trained model (presumably from `pepenc`) can be used to embed peptides using a command similar to
the following.

```
embed-peptides /prj/peptide-encoder-model/conf/base/config.yaml /prj/peptide-encoder-model/data/raw/sample-peptides.test.csv /prj/peptide-encoder-model/data/processed/embeddings.parquet --logging-level INFO
```

The `--help` flag can be used to see a description of all arguments to the script.

### Documentation

Unfortunately, there is no sphinx, etc., documentation at this time. The file `conf/base/config.yaml` shows examples of
all hyperparameters, data files, etc., for using training models.

#### Input data format

The models in this project require an input csv file that has one row which is a header and remaining rows which are
the peptides for the various datasets. The column in the csv file with the peptide sequences must be named `sequence`.
(This can be adjusted if calling the `pepenc_model` library from python code.)

#### Output data format

The embedded peptide sequences are saved as a data frame in parquet format. One column is named `peptide_sequence`, and
it contains the original amino acid sequence as a string. A second column named `embedded_sequence` contains the
embedding for the associated peptide as a 1-dimensional numpy array.

### Testing the code

The project uses `pytest` for unit testing. The testing prerequisites (though not other dependencies, as described
above) can be installed as follows.

```
pip3 install .[test]
```

After installing `pytest` and other testing dependencies, tests can be performed as follows.

```
cd /path/to/peptide-encoder-model
pytest .