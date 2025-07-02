import os
import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Test Text Generation")
    parser.add_argument(
        "--model_path",
        type=str,
        default="AIDC-AI/Ovis-U1-3B",
    )
    args = parser.parse_args()
    return args


def build_inputs(model, text_tokenizer, visual_tokenizer, prompt, pil_image):
    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, 
        [pil_image], 
        generation_preface='',
        return_labels=False,
        propagate_exception=False,
        multimodal_type='single_image',
        fix_sample_overall_length_navit=False
        )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    if pixel_values is not None:
        pixel_values = torch.cat([
            pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) if pixel_values is not None else None
        ],dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
        ],dim=0)
    return input_ids, pixel_values, attention_mask, grid_thws


def pipe_txt_gen(model, pil_image, prompt):
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    gen_kwargs = dict(
          max_new_tokens=4096,
          do_sample=False,
          top_p=None,
          top_k=None,
          temperature=None,
          repetition_penalty=None,
          eos_token_id=text_tokenizer.eos_token_id,
          pad_token_id=text_tokenizer.pad_token_id,
          use_cache=True,
      )
    prompt = "<image>\n" + prompt
    input_ids, pixel_values, attention_mask, grid_thws = build_inputs(model, text_tokenizer, visual_tokenizer, prompt, pil_image)
    with torch.inference_mode():
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)[0]
        gen_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
    return gen_text


def main():
    # load model
    args = parse_args()
    model, loading_info = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                torch_dtype=torch.bfloat16,
                                                output_loading_info=True,
                                                trust_remote_code=True
                                                )
    print(f'Loading info of Ovis-U1:\n{loading_info}')

    model = model.eval().to("cuda")
    model = model.to(torch.bfloat16)
    image_path = os.path.join(os.path.dirname(__file__), "docs", "imgs", "cat.png")
    pil_img = Image.open(image_path).convert('RGB')
    prompt = "What is it?"
    gen_txt = pipe_txt_gen(model, pil_img, prompt)
    print(gen_txt)


if __name__ == "__main__":
    main()
