from faker import Faker
from text import generate_paragraph as generate_paragraph_text, generate_title, generate_sentence
from pylatex import Document, Section, Subsection, Math, Axis, Figure, Alignat, Itemize, Enumerate, NoEscape, Tabular, Table, LongTable, Subsubsection, Command
from pylatex.utils import italic, bold

from tqdm import tqdm
from pathlib import Path
import numpy as np
import random
from datasets import load_dataset
from PIL import Image


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
        if random.random() < 0.70:
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

def generate_figure(doc, words, image_index, image_path):
    """
    Generate a random figure with random size and random pixel values.
    """
    # Random size, dimension too large will cause LaTeX to fail,
    width = random.randint(100, 300)
    height = width + random.randint(-50, 100)

    random_image_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    random_image = Image.fromarray(random_image_array)
    random_image.save(image_path)

    with doc.create(Figure(position="h!")) as fig:
        fig.add_image(str(image_path), width=f"{width}px")
        fig.add_caption(generate_sentence(words))

    image_index += 1
    return image_index

def generate_paragraph(doc, words, math_formulas):
    """
    Generate a random paragraph with random content, maybe including italics, bold text, or math.
    """
    doc.append(generate_paragraph_text(words, capitalize=True) + " ")

    # Randomly set italics, bold, inline math, or none of them to True
    choices = ["italics", "bold", "inline_math", None, "inline_math", "inline_math"]
    choice = random.choice(choices)

    # Create a random sentence length, maximum of 15 words, favoring shorter sentences
    sentence_length = min(np.random.geometric(0.5), 15)

    if choice == "italics":
        doc.append(italic(generate_sentence(words, 1, sentence_length)))
        doc.append(generate_paragraph_text(words))
    if choice == "bold":
        doc.append(bold(generate_sentence(words, 1, sentence_length)))
        doc.append(generate_paragraph_text(words))
    if choice == "inline_math":
        doc.append(NoEscape(generate_inline_math_expression(math_formulas)))
        doc.append(generate_paragraph_text(words))

    doc.append(NoEscape(r"\\\\"))

def generate_list(doc, words):
    """
    Generate a random list with random number of items.
    """
    with doc.create(Enumerate()) as enum:
        num_items = random.randint(2, 9)
        for i in range(num_items):
            enum.add_item(generate_sentence(words))

def generate_unnumbered_list(doc, words):
    """
    Generate a random unnumbered list with random number of items.
    """
    with doc.create(Itemize()) as itemize:
        num_items = random.randint(2, 9)
        for i in range(num_items):
            itemize.add_item(generate_sentence(words))

def generate_multi_line_math(doc, math_formulas, numbering):
    """
    Generate a random multi-line math expression with random number of equations.
    """
    with doc.create(Alignat(numbering=numbering, escape=False)) as agn:
        # Favor fewer equations, but allow up to 5
        num_equations = min(random.randint(1, 5), random.randint(1, 5))
        for i in range(num_equations):
            agn.append(NoEscape(generate_math_expression(math_formulas)))
            if num_equations > 1 and i < num_equations - 1:
                agn.append(r"\\")


def create_tex_file(words, math_formulas, short_math_formulas, document_name):
    """
    Create a random, synthetic .tex file with random text, math expressions, tables, etc.
    """
    fake = Faker()

    # Document settings
    document_class = random.choice(["article", "report", "book", "scrartcl", "scrreprt", "scrbook"])
    document_image_index = 0

    doc = Document(
        documentclass=document_class,
        inputenc="utf8",
        lmodern=True,
        textcomp=True,
    )

    section_numbering = random.choice([True, False])
    math_numbering = random.choice([True, False, False])

    doc.preamble.append(NoEscape(r"\usepackage[swedish]{babel}"))

    if random.random() < 0.1:
        doc.preamble.append(Command("title", generate_title(words)))
        doc.preamble.append(Command("author", fake.name()))
        doc.preamble.append(Command("date", fake.date()))
        doc.append(NoEscape(r"\maketitle"))
        if random.random() < 0.3:
            # Title page
            doc.append(NoEscape(r"\newpage"))

    # Generate a paragraph with 30% probability
    if random.random() < 0.3:
        generate_paragraph(doc, words, short_math_formulas)

    # Section
    with doc.create(Section(generate_title(words), numbering=section_numbering)):
        # Multi-line math is often preceded by a paragraph
        if random.random() < 0.9:
            generate_paragraph(doc, words, short_math_formulas)
        generate_multi_line_math(doc, math_formulas, math_numbering)
        # Multi-line math is often followed by a paragraph
        if random.random() < 0.9:
            generate_paragraph(doc, words, short_math_formulas)

        if random.random() < 0.5:
            generate_paragraph(doc, words, short_math_formulas)

        if random.random() < 0.5:
            generate_paragraph(doc, words, short_math_formulas)
            document_image_index = generate_figure(doc, words, document_image_index, f"{document_name}_{document_image_index}.png")
            if random.random() < 0.7:
                generate_paragraph(doc, words, short_math_formulas)

        # Subsection
        with doc.create(Subsection(generate_title(words), numbering=section_numbering)):
            if random.random() < 0.9:
                generate_paragraph(doc, words, short_math_formulas)
            generate_list(doc, words)
            if random.random() < 0.9:
                generate_paragraph(doc, words, short_math_formulas)

            generate_multi_line_math(doc, math_formulas, math_numbering)

            if random.random() < 0.5:
                generate_paragraph(doc, words, short_math_formulas)

            generate_unnumbered_list(doc, words)
            if random.random() < 0.9:
                generate_paragraph(doc, words, short_math_formulas)

            generate_table(doc, words)
            if random.random() < 0.9:
                generate_paragraph(doc, words, short_math_formulas)

            if random.random() < 0.5:
                generate_paragraph(doc, words, short_math_formulas)

            with doc.create(Subsubsection(generate_title(words), numbering=section_numbering)):
                if random.random() < 0.9:
                    generate_paragraph(doc, words, short_math_formulas)
                generate_multi_line_math(doc, math_formulas, math_numbering)
                if random.random() < 0.9:
                    generate_paragraph(doc, words, short_math_formulas)

                if random.random() < 0.5:
                    generate_table(doc, words)

                if random.random() < 0.5:
                    generate_paragraph(doc, words, short_math_formulas)

                if random.random() < 0.5:
                    generate_paragraph(doc, words, short_math_formulas)
                    document_image_index = generate_figure(doc, words, document_image_index, f"{document_name}_{document_image_index}.png")
                    generate_paragraph(doc, words, short_math_formulas)

                if random.random() < 0.1:
                    generate_list(doc, words)

    try:
        doc.generate_pdf(document_name, clean_tex=False)
    except UnicodeDecodeError as e:
        pass
    except Exception as e:
        print(e)


if __name__ == "__main__":
    lang = "swedish"
    # Get text, encoded in UTF-8
    with open(f"{lang}_text2.txt", "r", encoding="utf-8") as file:
        text = file.read()
        words = text.split()

    # Create necessary file structure
    # tex_dir = Path("tex")
    # tex_dir.mkdir(parents=True, exist_ok=True)

    saved_formulas = Path("math_formulas.npy").exists() and Path("short_math_formulas.npy").exists()
    if saved_formulas:
        # Get math formulas
        dataset = load_dataset("OleehyO/latex-formulas", "cleaned_formulas", split="train", columns=["latex_formula"])
        all_math_formulas = dataset["latex_formula"]

        # Use only a subset of the math formulas, skip the last 2000 formulas
        all_math_formulas = np.array(all_math_formulas)

        # Only use a subset of the math formulas: ones that begin with \begin{align*} and end with \end{align*}
        math_formulas = np.array([formula[14:-12] for formula in all_math_formulas[:-5000] if formula.startswith(r"\begin{align*}") and formula.endswith(r"\end{align*}")])
        # math_formulas = math_formulas[np.vectorize(len)(math_formulas) < 75]
        short_math_formulas = math_formulas[np.vectorize(len)(math_formulas) < 70]

        # Save math formula lists as numpy arrays
        np.save("math_formulas.npy", math_formulas)
        np.save("short_math_formulas.npy", short_math_formulas)

    # Load math formula lists from numpy arrays
    math_formulas = np.load("math_formulas.npy")
    short_math_formulas = np.load("short_math_formulas.npy")

    # Create documents
    num_documents = 900
    for i in tqdm(range(1, num_documents+1), desc="Creating documents"):
        create_tex_file(words, math_formulas, short_math_formulas, document_name=f"{i}")

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
