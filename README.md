# Individual Project - README

## Overview
This project explores the exciting field of AI poetry generation using deep learning techniques. We've developed a Seq2Seq model tailored for generating poetry in the Spanish language. This README will guide you through the setup and usage of our AI poetry generator.

## Prerequisites
Before you begin, make sure you have the following libraries installed:
- [torch](https://pytorch.org/)
- [torchtext](https://github.com/pytorch/text)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [spacy](https://spacy.io/)
- [tqdm](https://tqdm.github.io/)

You can install these libraries using pip:

```bash
pip install torch torchtext numpy pandas spacy tqdm
```

Additionally, ensure that you have a compatible GPU if you want to leverage GPU acceleration. We have included code to automatically switch between CPU and GPU based on availability. 

## Dataset
Our AI poetry generator is trained on a dataset of Spanish poems. You can access and download this dataset from the following link:

[Spanish Poetry Dataset](https://www.kaggle.com/datasets/andreamorgar/ spanish-poetry-dataset)

## Customization
You can customize the AI poetry generator by modifying various hyperparameters and settings in the code. These include the model architecture, learning rate, sequence length, and more. Dive into the code to explore and tweak these settings according to your preferences.

## Acknowledgments
This project leverages the power of deep learning and builds upon various open-source libraries and frameworks, including PyTorch and spaCy. We'd like to express our gratitude to the open-source community for their contributions.
