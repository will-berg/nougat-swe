import random
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pylatex import Document, Alignat, NoEscape
from datasets import load_dataset
from lorem.text import TextLorem


def generate_math_expression(math_formulas):
    """
    Generate a random math expression from the given list of math formulas.
    """
    formula = random.choice(math_formulas)
    return rf"{formula}"

def generate_multi_line_math(doc, math_formulas):
    """
    Generate a random multi-line math expression with random number of equations.
    """
    math_numbering = random.choice([False, False])
    with doc.create(Alignat(numbering=math_numbering, escape=False)) as agn:
        num_equations = random.randint(1, 10)
        for _ in range(num_equations):
            agn.append(NoEscape(generate_math_expression(math_formulas) + r"\\"))

def generate_inline_math_expression(math_formulas):
    """
    Generate a random math expression from the given list of math formulas.
    """
    formula = random.choice(math_formulas)
    # If the math expression ends with a period, remove it with a 90% probability
    if formula[-1] == "." and random.random() < 0.9:
        formula = formula[:-1]
    return rf"${formula}$ "

def generate_inline_math_paragraph(doc, short_math_formulas):
    min_words = random.randint(1, 5)
    max_words = random.randint(5, 15)
    lorem = TextLorem(wsep=" ", srange=(min_words, max_words))

    num_math = random.randint(5, 15)
    for _ in range(num_math):
        doc.append(lorem.sentence() + " ")
        doc.append(NoEscape(generate_inline_math_expression(short_math_formulas)))
        doc.append(lorem.sentence() + " ")

def generate_content(doc, math_formulas, choice):
    """
    Generate a random paragraph with random content, maybe including italics, bold text, or math.
    """
    if choice == "inline_math":
        generate_inline_math_paragraph(doc, math_formulas)
    if choice == "multi_line_math":
        generate_multi_line_math(doc, math_formulas)

def create_tex_file(math_formulas, short_math_formulas, document_name, text_type):
    """
    Create a random, synthetic .tex file with random text, math expressions, tables, etc.
    """
    # Document settings
    document_class = random.choice(["article", "report", "book", "scrartcl", "scrreprt", "scrbook"])
    doc = Document(
        documentclass=document_class,
        inputenc="utf8",
        lmodern=True,
        textcomp=True,
    )
    doc.preamble.append(NoEscape(r"\usepackage[swedish]{babel}"))

    if text_type == "inline_math":
        generate_content(doc, short_math_formulas, text_type)
    if text_type == "multi_line_math":
        generate_content(doc, math_formulas, text_type)

    try:
        doc.generate_pdf(document_name, clean_tex=False)
    except UnicodeDecodeError as e:
        pass
    except Exception as e:
        print(e)


if __name__ == "__main__":
    saved_formulas = Path("math_formulas.npy").exists() and Path("short_math_formulas.npy").exists()
    if not saved_formulas:
        # Get math formulas
        dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas", split="train", columns=["latex_formula"])
        all_math_formulas = dataset["latex_formula"]
        all_math_formulas = np.array(all_math_formulas)

        # Only use a subset of the math formulas: ones that begin with \begin{align*} and end with \end{align*}
        math_formulas = np.array([formula[14:-12] for formula in all_math_formulas[-2000:] if formula.startswith(r"\begin{align*}") and formula.endswith(r"\end{align*}")])
        short_math_formulas = math_formulas[np.vectorize(len)(math_formulas) < 50]
        math_formulas = math_formulas[np.vectorize(len)(math_formulas) < 100]
        print(f"Number of math formulas: {len(math_formulas)}")
        print(f"Number of short math formulas: {len(short_math_formulas)}")

        # Save math formula lists as numpy arrays
        np.save("math_formulas.npy", math_formulas)
        np.save("short_math_formulas.npy", short_math_formulas)

    # Load math formula lists from numpy arrays
    math_formulas = np.load("math_formulas.npy")
    short_math_formulas = np.load("short_math_formulas.npy")

    # Create documents
    num_documents = 1
    for i in tqdm(range(1, num_documents+1), desc="Creating documents"):
        # Cycle through the test types
        text_type = "multi_line_math"
        create_tex_file(math_formulas, short_math_formulas, document_name=f"{text_type}_{i}", text_type=text_type)

    # Remove all auxiliary files in the current directory
    for file in Path(".").iterdir():
        if file.suffix in [".aux", ".log", ".png"]:
            file.unlink()
