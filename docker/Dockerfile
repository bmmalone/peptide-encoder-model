FROM nvcr.io/nvidia/pytorch:21.12-py3

# Install dependencies (doing this first to have them cached).
COPY requirements.docker.txt /tmp/pepenc-model-requirements.txt
RUN pip install -r /tmp/pepenc-model-requirements.txt

COPY pepenc_model pepenc_model/pepenc_model
COPY setup.py pepenc_model
COPY README.md pepenc_model
RUN pip install -e pepenc_model