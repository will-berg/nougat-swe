"""
From Nougat:
In a scientific research article, there are three distinct types of text: 1) plain text, which
comprises the majority of the document, 2) mathematical expressions, and 3) tables.

In this script we will generate documents with these three types of text.
"""
from text import generate_paragraph as generate_paragraph_text, generate_sentence
from pylatex import Document, Math, Axis, Figure, Alignat, NoEscape, Tabular, Table, LongTable
from pylatex.utils import italic, bold

from tqdm import tqdm
from pathlib import Path
import numpy as np
import random
from datasets import load_dataset

def generate_word(words):
    start = random.randint(1, len(words) - 1)
    word = words[start]
    # Make the word bold or italic
    if random.random() < 0.15:
        word = italic(word)
    elif random.random() < 0.15:
        word = bold(word)
    return word

def generate_table(doc, words):
    """
    Generate a random table with random number of rows and columns, and random content.
    """
    num_rows = random.randint(2, 10)
    num_cols = random.randint(2, 7)

    table_data = np.empty((num_rows, num_cols), dtype=object)
    for i in range(num_rows):
        for j in range(num_cols):
            # Randomly choose between an integer, a float, and a word
            table_data[i][j] = random.choice([str(random.randint(-100, 500)), str(round(random.uniform(-100, 150), random.randint(1, 3))), generate_word(words)])

    with doc.create(Table(position="h!")) as table:
        # Centering
        if random.random() < 0.75:
            table.append(NoEscape(r"\centering"))
        column_styles = {
            1: "|".join(["c"] * num_cols),                       # c|c|c
            2: "|" + "|".join(["c"] * num_cols) + "|",           # |c|c|c|
            3: "|" + " ".join(["c"] * num_cols) + "|",           # |ccc|
            4: " ".join(["c"] * num_cols),                       # ccc
            5: "||" + " ".join(["c"] * num_cols) + "||",         # ||ccc||
            6: "||" + "|".join(["c"] * num_cols) + "||",           # ||c|c|c||
        }
        column_num = random.choice(list(column_styles.keys()))
        column_style = column_styles[column_num]
        row_style = random.randint(1, 5)
        with doc.create(Tabular(column_style)) as tabular:
            if row_style == 1:
                tabular.add_hline()
                for row in table_data:
                    tabular.add_row(row)
                    tabular.add_hline()
            if row_style == 2:
                for row in table_data:
                    tabular.add_row(row)
            if row_style == 3:
                tabular.add_hline()
                for row in table_data:
                    tabular.add_row(row)
                tabular.add_hline()
            # Horizontal lines at the top, after the first row, and bottom
            if row_style == 4:
                tabular.add_hline()
                for i, row in enumerate(table_data):
                    if i == 1:
                        tabular.add_hline()
                    tabular.add_row(row)
                tabular.add_hline()
            # No lines
            if row_style == 5:
                for row in table_data:
                    tabular.add_row(row)
        table.add_caption(generate_sentence(words))

def generate_math_expression(math_formulas):
    """
    Generate a random math expression from the given list of math formulas.
    """
    formula = random.choice(math_formulas)
    return rf"{formula}"

def generate_inline_math_expression(math_formulas):
    """
    Generate a random math expression from the given list of math formulas.
    """
    formula = random.choice(math_formulas)
    return rf"${formula}$ "

def generate_multi_line_math(doc, math_formulas):
    """
    Generate a random multi-line math expression with random number of equations.
    """
    # math_numbering = random.choice([True, False])
    with doc.create(Alignat(numbering=False, escape=False)) as agn:
        num_equations = random.randint(1, 6)
        for i in range(num_equations):
            agn.append(NoEscape(generate_math_expression(math_formulas)))
            if num_equations > 1 and i < num_equations - 1:
                agn.append(r"\\")

def generate_content(doc, words, math_formulas, choice):
    """
    Generate a random paragraph with random content, maybe including italics, bold text, or math.
    """
    if choice == "inline_math":
        doc.append(generate_paragraph_text(words, capitalize=True) + " ")
        doc.append(NoEscape(generate_inline_math_expression(math_formulas)))
        doc.append(generate_paragraph_text(words))
    if choice == "multi_line_math":
        if random.random() < 0.7:
            doc.append(generate_paragraph_text(words, capitalize=True) + " ")
        generate_multi_line_math(doc, math_formulas)
        if random.random() < 0.7:
            doc.append(generate_paragraph_text(words))
    if choice == "table":
        if random.random() < 0.7:
            doc.append(generate_paragraph_text(words, capitalize=True) + " ")
        generate_table(doc, words)
        if random.random() < 0.7:
            doc.append(generate_paragraph_text(words))

def create_tex_file(words, math_formulas, short_math_formulas, document_name, text_type):
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

    doc.preamble.append(NoEscape(r"\usepackage[swedish]{babel}" + "\n"))
    # Choose a random font
    if random.random() < 0.75:
        font = random.choice(["helvet", "avant", "palatino", "times", "charter"])
        doc.preamble.append(NoEscape(rf"\usepackage{{{font}}}" + "\n"))

    if text_type == "inline_math":
        generate_content(doc, words, short_math_formulas, text_type)
    if text_type == "multi_line_math":
        generate_content(doc, words, math_formulas, text_type)
    if text_type == "table":
        generate_content(doc, words, math_formulas, text_type)

    try:
        doc.generate_pdf(document_name, clean_tex=False)
    except UnicodeDecodeError as e:
        pass
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Get text, encoded in UTF-8
    with open(f"swedish_text.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words = text.split()

    saved_formulas = Path("math_formulas.npy").exists() and Path("short_math_formulas.npy").exists()
    if not saved_formulas:
        # Get math formulas
        dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas", split="train", columns=["latex_formula"])
        all_math_formulas = dataset["latex_formula"]
        all_math_formulas = np.array(all_math_formulas)

        # Only use a subset of the math formulas: ones that begin with \begin{align*} and end with \end{align*}
        math_formulas = np.array([formula[14:-12] for formula in all_math_formulas[:-5000] if formula.startswith(r"\begin{align*}") and formula.endswith(r"\end{align*}")])
        short_math_formulas = math_formulas[np.vectorize(len)(math_formulas) < 75]

        # Save math formula lists as numpy arrays
        np.save("math_formulas.npy", math_formulas)
        np.save("short_math_formulas.npy", short_math_formulas)

    # Load math formula lists from numpy arrays
    math_formulas = np.load("math_formulas.npy")
    short_math_formulas = np.load("short_math_formulas.npy")

    # Create documents
    num_documents = 18_000
    for i in tqdm(range(1, num_documents+1), desc="Creating documents"):
        text_type = random.choice(["table", "inline_math", "multi_line_math"])
        create_tex_file(words, math_formulas, short_math_formulas, document_name=f"{text_type}_{i}", text_type=text_type)

    # Remove all auxiliary files in the current directory
    for file in Path(".").iterdir():
        if file.suffix in [".aux", ".log", ".png"]:
            file.unlink()

    # Move all tex and pdf files to the tex and pdf directories in nougat_prereqs
    tex_source = Path("nougat_prereqs/tex_source")
    pdfs = Path("nougat_prereqs/pdfs")
    for file in Path(".").iterdir():
        if file.suffix == ".tex":
            file.replace(tex_source / file.name)
        if file.suffix == ".pdf":
            file.replace(pdfs / file.name)