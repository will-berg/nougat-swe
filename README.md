## Data
The dataset is available on Huggingface.
The code used to generate the data is available in the `data/code` directory.

## Model
The model related code is available in the `model` directory.
This includes code for fine-tuning, testing, inference, etc.

**Prerequisites for prediction:**
* Working cuda installation, see for example: https://pytorch.org/get-started/locally/
* Requirements listed in `model/predict/requirements.txt`

Run prediction on a pdf with the following command: `python model/predict/predict.py pdf_path` and it will produce an output `.mmd` file in the current directory.

**Links:**
* The model itself is available on Huggingface: https://huggingface.co/powow/nougat-swe
* Training data is available on Huggingface: https://huggingface.co/datasets/powow/swe-data
