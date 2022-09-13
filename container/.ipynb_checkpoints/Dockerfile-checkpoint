FROM 636218042492.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-detectron2:latest

RUN pip3 install --upgrade datasets==2.2.2 pathos seqeval

# SageMaker configs
ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# Copies the training code inside the container
COPY train.py /opt/ml/code/train.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py
