import subprocess
from pylatex import Document, NoEscape, Alignat, Math
from nougat.dataset.rasterize import rasterize_paper
from pathlib import Path
import json

def create_tex_file(document_name, formula):
    """
    Create synthetic .tex file with one math formula.
    """
    # Document settings
    doc = Document()

    # Add the math formula
    # with doc.create(Alignat(numbering=False, escape=False)) as agn:
        # agn.append(NoEscape(f"${formula}$"))
    with doc.create(Math(data=[NoEscape(formula)])):
        pass

    try:
        doc.generate_pdf(document_name, clean_tex=False)
    except UnicodeDecodeError as e:
        pass
    except Exception as e:
        print(e)

def create_images(images_path, pdfs_path):
    """
    Rasterize the pdfs and save the page image for a document in the images directory.
    Saved in format "images/documentName/pageNumber.png"
    """
    for pdf in pdfs_path.iterdir():
        if pdf.suffix == ".pdf":
            outpath = images_path / pdf.stem
            # If the pdf has not been rasterized yet, rasterize it
            if not outpath.exists():
                outpath.mkdir(parents=True, exist_ok=True)
                rasterize_paper(pdf=pdf, outpath=outpath, dpi=96)

def create_dataset(images_path, markdown_path):
    """
    Combine the image paths and the corresponding markdown content into a jsonl file
    with the format {"image": "path/to/image.png", "markdown": "markdown content of the page", "meta": "[]"}
    (run after .mmd and .png files are created)
    """
    # Create the jsonl dataset file in ../datasets/swe_text.jsonl
    with open("math_formulas.jsonl", "w") as f:
        for image_folder, markdown_folder in zip(images_path.iterdir(), markdown_path.iterdir()):
            images = sorted(image_folder.iterdir())
            markdowns = sorted(markdown_folder.iterdir())
            # Check that the folder names match
            try:
                assert image_folder.name == markdown_folder.name
            except AssertionError:
                print(f"Folder names do not match: {image_folder.name} and {markdown_folder.name}")
                continue
            for image, markdown in zip(images, markdowns):
                # Check that they have the same name
                assert image.stem == markdown.stem
                data_sample = {}
                md_path = Path(markdown)
                data_sample["image"] = f"{images_path}/{image_folder.name}/{image.name}"
                data_sample["markdown"] = md_path.read_text(encoding="utf8").strip()
                data_sample["meta"] = "[]"
                f.write(json.dumps(data_sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    """
    with open("test_formulas") as f:
        indices = f.readlines()
    # Each formula in formulas is a line number that points to a formula in the im2latex_formulas.lst file
    with open("im2latex_formulas", encoding="ISO-8859-1") as f:
        formulas = f.readlines()
    for index in indices:
        index = index.strip()
        formula = str(formulas[int(index)]).strip()
        # Create folder with test_{index} as the folder name
        folder_name = Path("markdown") / f"test_{index}"
        folder_name.mkdir(parents=True, exist_ok=True)
        formula = rf"\[{formula}\]"
        # Create a 01.mmd file with the formula as the content
        with open(folder_name / "01.mmd", "w") as f:
            f.write(formula)
        # create_tex_file(f"test_{index}".strip(), formula)
    """

    # Move all .pdf files and .tex files to the pdf and tex directories, using subprocess
    # subprocess.run("mv *.pdf pdf", shell=True)
    # subprocess.run("mv *.tex tex", shell=True)

    # Create images from the .pdf files
    # create_images(Path("images"), Path("pdf"))
    create_dataset(Path("images"), Path("markdown"))
