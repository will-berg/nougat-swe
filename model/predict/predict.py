from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
import sys
from util import rasterize_paper, StoppingCriteriaScores
from PIL import Image
from transformers import StoppingCriteriaList
from pathlib import Path


def load_model_and_processor(repo="powow/nougat-swe"):
    processor = NougatProcessor.from_pretrained(repo)
    model = VisionEncoderDecoderModel.from_pretrained(repo)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model, processor, device


if __name__ == "__main__":
    model, processor, device = load_model_and_processor()
    model.to(device)

    pdf = sys.argv[1]
    images = rasterize_paper(pdf=pdf, return_pil=True)
    images = [Image.open(image) for image in images]
    pixel_values = processor(images=images, return_tensors="pt", padding=True).pixel_values.to(device)

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

    # Save output in current directory
    with open(f"{Path(pdf).stem}.mmd", "w") as f:
        f.write("\n".join(generated))
