"""
Use the same preprocessing steps as outlined here: https://huggingface.co/datasets/OleehyO/latex-formulas
"""
import re


def clean_formulas():
	# All the formulas are in the file test_formulas.txt, one formula per line
	with open("test_formulas.txt", "r") as f:
		formulas = f.readlines()

	# Remove the unwanted content using regex
	cleaned_formulas = []
	for formula in formulas:
		# Remove the content
		formula = re.sub(r"\\label{.*?}", "", formula)
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
		cleaned_formulas.append(formula)
		# textstyle, displaystyle, box stuff (e.g., fbox, mbox, hbox, etc.)

	# Write the cleaned formulas back to the file
	with open("cleaned_formulas.txt", "w") as f:
		for formula in cleaned_formulas:
			f.write(formula)


if __name__ == "__main__":
	clean_formulas()

