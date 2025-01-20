from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
from util import StoppingCriteriaScores
from PIL import Image
from transformers import StoppingCriteriaList
from pathlib import Path
from metrics import compute_metrics
import os
import numpy as np
import re


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

def load_model_and_processor(repo="powow/nougat-swe"):
    processor = NougatProcessor.from_pretrained(repo)
    model = VisionEncoderDecoderModel.from_pretrained(repo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, processor, device

def load_images_and_ground_truth(image_dir, markdown_dir):
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
            data_pairs.append((image_path, ground_truth))
        else:
            print(f"Skipping folder {folder}: Expected one .png and one .mmd file.")

    return data_pairs


"""
The code below is for testing prediction with the nougat-swe model. It will be compared to prediction with
the TexTeller model, which is ran from that model's directory using the following command in the src directory:
python inference.py -img "img_path.png" --inference-mode cuda

A separate main function will have to be created for the TexTeller model.
The final results produced by metrics.py will then be compared.
"""
if __name__ == "__main__":
    image_gt_pairs = load_images_and_ground_truth(Path("images"), Path("markdown"))
    print(f"Loaded {len(image_gt_pairs)} image-ground truth pairs.")

    # Split the pairs into batches of size 20
    batch_size = 10
    image_gt_pairs = [image_gt_pairs[i:i + batch_size] for i in range(0, len(image_gt_pairs), batch_size)]

    model, processor, device = load_model_and_processor()
    model.to(device)

    results = []

    for batch in image_gt_pairs:
        images = [Image.open(image_path) for image_path, _ in batch]
        ground_truths = [ground_truth for _, ground_truth in batch]

        # Process the images
        pixel_values = processor(images=images, return_tensors="pt", padding=True).pixel_values.to(device)

        # Generate the predictions
        outputs = model.generate(pixel_values,
                                  min_length=1,
                                  max_length=3584,
                                  bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                  return_dict_in_generate=True,
                                  output_scores=True,
                                  stopping_criteria=StoppingCriteriaList([StoppingCriteriaScores()]),
        )

        generated = processor.batch_decode(outputs[0], skip_special_tokens=True)
        generated = processor.post_process_generation(generated, fix_markdown=False)

        metrics = compute_metrics(generated, ground_truths)
        # Print the average metrics for this batch
        print(metrics)
        results.append(metrics)

    # Save the results as a npy file
    np.save("nougat_results.npy", results)

    # Results contains a list of dictionaries, one for each batch.
    # Take the mean of each metric across all batches.
    aggregated_results = {metric: np.nanmean([result[metric] for result in results]) for metric in results[0]}
    print(aggregated_results)
