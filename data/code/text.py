import random

def capitalize_first_letter(s):
    return s[0].upper() + s[1:]

def generate_paragraph(words, capitalize=False):
	start = random.randint(0, len(words))
	length = random.randint(75, 125)
	if start + length > len(words):
		start = len(words) - length
	end = start + length

	paragraph = ' '.join(words[start:end])
	if capitalize:
		paragraph = capitalize_first_letter(paragraph)
	paragraph = paragraph + '.'
	return paragraph

def generate_title(words, lang="swe"):
	start = random.randint(0, len(words))
	length = random.randint(1, 5)
	if start + length > len(words):
		start = len(words) - length
	end = start + length

	title = ' '.join(words[start:end])
	# Capitalize the first letter of the title, remove all non-alphanumeric characters except for spaces
	if lang == "swe":
		title = title.capitalize()
	if lang == "eng":
		title = title.title()
	title = ''.join(e for e in title if e.isalnum() or e.isspace())
	return title

def generate_word(words):
	start = random.randint(0, len(words))
	word = words[start]
	return word

def generate_sentence(words, l=5, r=15):
	start = random.randint(0, len(words))
	length = random.randint(l, r)
	if start + length > len(words):
		start = len(words) - length
	end = start + length

	sentence = ' '.join(words[start:end])
	sentence = capitalize_first_letter(sentence) + " "
	return sentence
