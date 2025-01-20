from pathlib import Path
from metrics import compute_metrics
import os
import numpy as np
import re

import cv2 as cv
from texteller.src.models.ocr_model.utils.to_katex import to_katex
from texteller.src.models.ocr_model.utils.inference import inference as latex_inference
from texteller.src.models.ocr_model.model.TexTeller import TexTeller


def clean_formula(formula):
    formula = re.sub(r"\\label{.*?}", "", formula)
    formula = re.sub(r"\\label {.*?}", "", formula)
    formula = re.sub(r"%", "", formula)
    formula = re.sub(r"\\quad", "", formula)
    formula = re.sub(r"\\qquad", "", formula)
    formula = re.sub(r"\\vspace{.*?}", "", formula)
    formula = re.sub(r"\\hspace{.*?}", "", formula)
    formula = re.sub(r"\\parbox{.*?}", "", formula)
    formula = re.sub(r"\\hfill", "", formula)
    formula = re.sub(r"\\vfill", "", formula)
    formula = re.sub(r"\\tag{.*?}", "", formula)
    formula = re.sub(r"\\text{.*?}", "", formula)
    formula = re.sub(r"\\nonumber", "", formula)
    return formula

def load_images_and_ground_truth(image_dir, markdown_dir):
    """
    Load the images and ground truth from the specified directories.
    pairs: image_path, ground_truth (string)
    """
    data_pairs = []

    # Iterate over folders in the image directory
    for folder in os.listdir(image_dir):
        image_folder = image_dir / folder
        markdown_folder = markdown_dir / folder

        # Each folder should contain one image and one markdown file
        image_files = list(image_folder.glob("*.png"))
        markdown_files = list(markdown_folder.glob("*.mmd"))

        if len(image_files) == 1 and len(markdown_files) == 1:
            assert image_folder.name == markdown_folder.name
            image_path = image_files[0]
            with open(markdown_files[0], "r", encoding="utf-8") as f:
                ground_truth = f.read()
            # Before cleaning
            if re.search(r"\\qquad{.+}", ground_truth):
                continue
            ground_truth = clean_formula(ground_truth)
            # After cleaning
            if len(ground_truth) > 200:
                continue
            # Remove starting and ending equation delimiters: \[ and \]
            if ground_truth.startswith("\\[") and ground_truth.endswith("\\]"):
                ground_truth = ground_truth[2:-2]
            data_pairs.append((image_path, ground_truth))
        else:
            print(f"Skipping folder {folder}: Expected one .png and one .mmd file.")

    return data_pairs

def inference(img_path, model, tokenizer, inference_mode="cuda", num_beam=1):
    """
    TexTeller inference function.
    """
    img = cv.imread(img_path)
    res = latex_inference(model, tokenizer, [img], inference_mode, num_beam)
    res = to_katex(res[0])
    return res


if __name__ == "__main__":
    """
    image_gt_pairs = load_images_and_ground_truth(Path("images"), Path("markdown"))

    # Load the TexTeller model and tokenizer
    latex_rec_model = TexTeller.from_pretrained()
    tokenizer = TexTeller.get_tokenizer()

    results = []
    # Run TexTeller on the images and save the results
    for img_path, gt in image_gt_pairs:
        inference_result = inference(img_path, latex_rec_model, tokenizer)
        result = compute_metrics([inference_result], [gt])
        print(result)
        results.append(result)

    # Save the results as a npy file
    np.save("texteller_results.npy", results)

    # Results contains a list of dictionaries, one for each batch.
    # Take the mean of each metric across all batches.
    aggregated_results = {metric: np.mean([result[metric] for result in results]) for metric in results[0]}
    print(aggregated_results)
    """
    metric_results = np.load("texteller_results.npy", allow_pickle=True)
    # Get the average metric result for each metric, across all batches
    aggregated_results = {metric: np.nanmean([result[metric] for result in metric_results]) for metric in metric_results[0].keys()}
    print(aggregated_results)
    metric_results_nougat = np.load("nougat_results.npy", allow_pickle=True)
    aggregated_results_nougat = {metric: np.nanmean([result[metric] for result in metric_results_nougat]) for metric in metric_results_nougat[0].keys()}
    print(aggregated_results_nougat)


