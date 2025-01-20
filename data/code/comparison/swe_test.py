"""
Running various OCR tools on a Swedish text to compare their performance.
"""
import os
import json
from pathlib import Path
from metrics import compute_metrics
import numpy as np

import easyocr

from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor


if __name__ == "__main__":
    """
    # EasyOCR
    reader = easyocr.Reader(["sv"])
    # Surya
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    easyocr_results = []
    surya_results = []
    # Loop through swe_text.jsonl
    with open("swe_text.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        image = data["image"]
        ground_truth = data["markdown"]

        # EasyOCR
        easyocr_result = reader.readtext(image, detail=0, paragraph=True)
        easyocr_result = " ".join(easyocr_result)

        # Surya
        predictions = run_ocr([Image.open(image)], [["sv"]], det_model, det_processor, rec_model, rec_processor)
        predictions = predictions[0].text_lines
        surya_result = " ".join([line.text for line in predictions])

        # Calculate metrics
        easyocr_metrics = compute_metrics(easyocr_result, ground_truth)
        surya_metrics = compute_metrics(surya_result, ground_truth)
        easyocr_results.append(easyocr_metrics)
        surya_results.append(surya_metrics)
        print(f"EasyOCR: {easyocr_metrics}")
        print(f"Surya: {surya_metrics}")
        print()

    # Save the results
    np.save("easyocr_results.npy", easyocr_results)
    np.save("surya_results.npy", surya_results)
    """

    # Take the mean of each metric across all samples, for both tools, each sample is a dictionary
    # with the metrics as keys.
    easyocr_results = np.load("easyocr_results.npy", allow_pickle=True)
    surya_results = np.load("surya_results.npy", allow_pickle=True)
    easyocr_aggregated_results = {}
    # Iterate over each metric/key in the first dictionary of the list
    for metric in easyocr_results[0].keys():
        for result in easyocr_results:
            if result[metric] is None:
                result[metric] = np.nan
        # Extract the values of this metric from all dictionaries in the list
        metric_values = [result[metric] for result in easyocr_results]
        # Compute the mean of the metric values, ignoring NaNs
        metric_mean = np.nanmean(metric_values)
        # Add the computed mean to the aggregated results dictionary
        easyocr_aggregated_results[metric] = metric_mean

    surya_aggregated_results = {metric: np.nanmean([result[metric] for result in surya_results]) for metric in surya_results[0].keys()}
    print("EasyOCR aggregated results:")
    print(easyocr_aggregated_results)
    print("Surya aggregated results:")
    print(surya_aggregated_results)





# Doctr
# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
#
# # Set swedish as the language
# model = ocr_predictor(pretrained=True, lang="sv")
# doc = DocumentFile.from_images("01.png")
# result = model(doc)
# print(result)


# pyocr
# import pyocr
# import pyocr.builders
# from PIL import Image
#
# tools = pyocr.get_available_tools()
# print(tools)
# tool = tools[0]
# langs = tool.get_available_languages()
# lang = langs[3]
# print(lang)
# txt = tool.image_to_string(
# 	Image.open("01.png"),
# 	lang=lang,
# 	builder=pyocr.builders.TextBuilder()
# )
# print(txt)
#

# from paddleocr import PaddleOCR
# # to switch the language model in order.
# ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
# img_path = '01.png'
# result = ocr.ocr(img_path, cls=True)
# for line in result:
#     print(line)
#

"""
# Go through all the swe_text markdown folders and save the ones that have only one .mmd file
from pathlib import Path

folder = Path("swe_text/markdown")
to_use = []

for subfolder in folder.iterdir():
    if len(list(subfolder.glob("*.mmd"))) == 1:
        to_use.append(subfolder.stem)

# Now get the ground truth from the corresponding txt files in the txt folder
folder = Path("swe_text/txt")

ground_truths = {}
for subfolder in to_use:
    txt_file = folder / f"{subfolder}.txt"
    with open(txt_file, "r", encoding="utf-8") as f:
        ground_truth = f.read()
        ground_truths[subfolder] = ground_truth

# Get corresponding images in the images folder
folder = Path("swe_text/images")

images = {}
for subfolder in to_use:
    image_file = folder / f"{subfolder}.png"
    images[subfolder] = image_file

# Combine the data
data = []
for key in ground_truths:
    data.append((images[key], ground_truths[key]))

print(data)
"""
