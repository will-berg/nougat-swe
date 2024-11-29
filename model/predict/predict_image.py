from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch

processor = NougatProcessor.from_pretrained("powow/nougat-swe")
model = VisionEncoderDecoderModel.from_pretrained("powow/nougat-swe")
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

image_path = "test.png"
image = Image.open(image_path)
pixel_values = processor(image, return_tensors="pt").pixel_values

outputs = model.generate(
	pixel_values.to(device),
	min_length=1,
	max_new_tokens=3584,
	bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)

print(sequence)
