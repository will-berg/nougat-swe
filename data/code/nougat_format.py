# Assumes that the PDF files are in the nougat_prereqs/pdfs directory and the LaTeX files are in the nougat_prereqs/tex_source directory.
import os
import subprocess
from pathlib import Path

pdf_path = "nougat_prereqs/pdfs/"
figure_path = "nougat_prereqs/figures/"
tex_path = "nougat_prereqs/tex_source/"
html_path = "nougat_prereqs/htmls/"
output_path = "../datasets/multi_math_test/"

def create_figures():
    """
    Use pdffigures2 to extract figures from PDF files and place the resulting
    json files in the nougat_prereqs/figures directory.
    """
    for pdf_file in Path(pdf_path).iterdir():
        if pdf_file.suffix == ".pdf":
            # subprocess.run(["java", "-jar", "pdffigures2.jar", "-d", figure_path, pdf_file])
            with open(Path(figure_path) / f"{pdf_file.stem}.json", "w") as file:
                file.write("[]")

def create_htmls():
    # Start the Docker container in detached mode
    container_id = os.popen(f"docker run -d -v \"{os.getcwd()}\":/workdir -w /workdir arxivvanity/engrafo tail -f /dev/null").read().strip()

    try:
        for tex_file in Path(tex_path).iterdir():
            if tex_file.suffix == ".tex":
                name = tex_file.stem
                subprocess.run(["docker", "exec", container_id, "engrafo", tex_file, html_path])
                os.rename(Path(html_path) / "index.html", Path(html_path) / f"{name}.html")
    finally:
        subprocess.run(["docker", "stop", container_id])

def create_paired_dataset():
    subprocess.run(["python", "../../nougat_fork/nougat/dataset/split_htmls_to_pages.py", "--html", html_path, "--pdfs", pdf_path, "--out", output_path, "--figure", figure_path, "--recompute"])
    subprocess.run(["python", "../../nougat_fork/nougat/dataset/create_index.py", "--dir", output_path, "--out", "../datasets/multi_math_test.jsonl"])

if __name__ == "__main__":
    # Create the folders if they don't exist
    Path(figure_path).mkdir(parents=True, exist_ok=True)
    Path(html_path).mkdir(parents=True, exist_ok=True)
    Path(output_path).mkdir(parents=True, exist_ok=True)

    print("Extracting figures...")
    create_figures()
    print("Convert .tex to .html...")
    create_htmls()
    print("Creating dataset...")
    create_paired_dataset()
