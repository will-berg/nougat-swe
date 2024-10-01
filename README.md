# nougat-swe
Fine-tuned version of nougat, available here: https://github.com/facebookresearch/nougat

## Data
The dataset is available on Huggingface.
The code used to generate the data is available in the `data/code` directory.
Training data is available on Huggingface: https://huggingface.co/datasets/powow/swe-data

## Model
The model related code is available in the `model` directory.
This includes code for fine-tuning, testing, inference, etc. 
The model itself is available on Huggingface: https://huggingface.co/powow/nougat-swe

**Prerequisites for prediction:**
* Working cuda installation, see for example: https://pytorch.org/get-started/locally/
* Requirements listed in `model/predict/requirements.txt`

Run prediction on a pdf with the following command: `python model/predict/predict.py pdf_path` and it will produce an output `.mmd` file in the current directory.
