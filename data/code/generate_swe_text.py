from datasets import load_dataset
import re
import os
import json
import subprocess
from pathlib import Path
from PIL import Image
import pytesseract
import random
from concurrent.futures import ThreadPoolExecutor

from nougat.dataset.rasterize import rasterize_paper


def create_text_files(ds, num_samples, text_path, lang="sv"):
    for i, sample in enumerate(ds["train"]):
        if i == num_samples:
            break
        content = sample["text"]
        # Remove recurring phrases from typical wiki articles: "Se även", "Källor", "Externa länkar", "Vidare läsning", "Noter"
        if lang == "sv":
            content = re.sub(r"Se även", "", content)
            content = re.sub(r"Källor", "", content)
            content = re.sub(r"Referenser", "", content)
            content = re.sub(r"Externa länkar", "", content)
            content = re.sub(r"Vidare läsning", "", content)
            content = re.sub(r"Noter", "", content)
        if lang == "en":
            content = re.sub(r"See also", "", content)
            content = re.sub(r"References", "", content)
            content = re.sub(r"External links", "", content)
            content = re.sub(r"Further reading", "", content)
            content = re.sub(r"Notes", "", content)

        content = re.sub(r"²", "", content)
        content = re.sub(r"³", "", content)

        extension = "txt"
        with open(f"{text_path}/{sample["id"]}.{extension}", "w") as f:
            f.write(content)

def convert_to_pdf(input_file, output_path):
    """
    Convert text file to PDF using pandoc.
    """
    # Randomize pandoc flag options documentclass and papersize to get a diverse dataset
    documentclass = random.choice(["article", "report", "book", "scrartcl", "scrreprt", "scrbook", "memoir"])
    # papersize = random.choice(["a4", "a5", "letter", "legal", "executive"])
    output_file = output_path / (input_file.stem + ".pdf")
    subprocess.run(["pandoc", input_file, "-o", output_file, f"--variable=documentclass={documentclass}"])

def create_images(images_path, pdfs_path):
    """
    Rasterize the pdfs and save the page images for a document in the images directory.
    Saved in format "images/documentName/pageNumber.png"
    """
    for pdf in pdfs_path.iterdir():
        if pdf.suffix == ".pdf":
            outpath = images_path / pdf.stem
            # If the pdf has not been rasterized yet, rasterize it
            if not outpath.exists():
                outpath.mkdir(parents=True, exist_ok=True)
                rasterize_paper(pdf=pdf, outpath=outpath, dpi=96)

def extract_text_from_png(image_file, outpath, lang="swe"):
    """
    Extract the text from the pages of the document (represented as .png images)
    and save it in a .mmd file.
    """
    with Image.open(image_file) as img:
        text = pytesseract.image_to_string(img, lang=lang)
    text = re.sub(r"\n+\s+?([^\s])", r"\n\n\1", text).strip()
    if not outpath.exists():
        with outpath.open("w") as f:
            f.write(text)

def create_markdown_files(markdown_path, images_path, lang="swe"):
    """
    Do the same for the markdown content that corresponds to the images
    Saves the markdown content for a document in the markdown directory with the format "markdown/documentName/pageNumber.mmd"
    """
    for folder in images_path.iterdir():
        if folder.is_dir():
            markdown_folder = markdown_path / folder.name
            markdown_folder.mkdir(parents=True, exist_ok=True)
            for file in folder.iterdir():
                if file.suffix == ".png":
                    outpath = markdown_folder / f"{file.stem}.mmd"
                    extract_text_from_png(file, outpath, lang=lang)

def create_dataset(images_path, markdown_path):
    """
    Combine the image paths and the corresponding markdown content into a jsonl file
    with the format {"image": "path/to/image.png", "markdown": "markdown content of the page", "meta": "[]"}
    (run after .mmd and .png files are created)
    """
    # Create the jsonl dataset file in ../datasets/swe_text.jsonl
    with open("../datasets/swe_text.jsonl", "w") as f:
        for image_folder, markdown_folder in zip(images_path.iterdir(), markdown_path.iterdir()):
            images = sorted(image_folder.iterdir())
            markdowns = sorted(markdown_folder.iterdir())
            # Check that the folder names match
            assert image_folder.name == markdown_folder.name
            for image, markdown in zip(images, markdowns):
                # Check that they have the same name
                assert image.stem == markdown.stem
                data_sample = {}
                md_path = Path(markdown)
                data_sample["image"] = f"{images_path}/{image_folder.name}/{image.name}"
                data_sample["markdown"] = md_path.read_text(encoding="utf8").strip()
                data_sample["meta"] = "[]"
                f.write(json.dumps(data_sample, ensure_ascii=False) + "\n")

def convert_text_files_to_pdfs(text_path, pdfs_path):
    """
    Convert all text files in the given directory to PDFs in parallel.
    """
    # List all text files in the directory
    text_files = [file for file in text_path.iterdir() if file.suffix == ".txt"]

    print("Converting text files to PDF...")

    # Use ThreadPoolExecutor to parallelize the conversion
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_to_pdf, file, pdfs_path) for file in text_files]

        # Wait for all futures to complete
        for future in futures:
            future.result()


if __name__ == "__main__":
    root_dir = Path("../datasets/swe_text")
    pdfs_path = root_dir / "pdf"
    markdown_path = root_dir / "markdown"
    text_path = root_dir / "txt"
    images_path = root_dir / "images"

    # Create directories if they don't exist
    root_dir.mkdir(parents=True, exist_ok=True)
    pdfs_path.mkdir(parents=True, exist_ok=True)
    markdown_path.mkdir(parents=True, exist_ok=True)
    text_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    lang = "sv"
    # ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}")

    # Create text files from the dataset
    # print("Creating text files...")
    # create_text_files(ds, num_samples=2500, text_path=text_path, lang)

    # Convert text files to PDFs
    convert_text_files_to_pdfs(text_path, pdfs_path)

    # Create images from PDFs
    print("Creating images from PDFs...")
    create_images(images_path, pdfs_path)

    # Create markdown files
    print("Creating markdown files from images...")
    create_markdown_files(markdown_path, images_path)

    # Create the dataset
    print("Creating jsonl dataset...")
    create_dataset(images_path, markdown_path)

    # Combine x text files and remove whitespace to create a single text file with the content of x articles
    if lang == "sv":
        file_path = "swedish_text.txt"
    else:
        file_path = "english_text.txt"
    x = 2000
    text_files = [file for file in text_path.iterdir() if file.suffix == ".txt"]
    with open(file_path, "w") as f:
        for file in text_files:
            if x == 0:
                break
            with open(file, "r") as f2:
                content = f2.read()
                # Remove whitespace
                content = re.sub(r"\s+", " ", content)
                f.write(content)
                x -= 1
