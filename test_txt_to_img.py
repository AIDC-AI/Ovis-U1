import os
import argparse
import math
import torch
from PIL import Image
from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description="Test Text-to-Image")
    parser.add_argument(
        "--model_path",
        type=str,
        default="AIDC-AI/Ovis-U1-3B",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--steps", type=int, default=50,
    )
    parser.add_argument(
        "--txt_cfg", type=float, default=5,
    )
    args = parser.parse_args()
    return args


def load_blank_image(width, height):
    pil_image = Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')
    return pil_image

def build_inputs(model, text_tokenizer, visual_tokenizer, prompt, pil_image, target_width, target_height):
    if pil_image is not None:
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(pil_image, target_size)
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
        pil_image = pil_image.resize((resized_width, resized_height))
    else:
        vae_pixel_values = None
        cond_img_ids = None

    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, 
        [pil_image], 
        generation_preface=None,
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
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values


def pipe_t2i(model, prompt, height, width, steps, cfg, seed=42):
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    gen_kwargs = dict(
          max_new_tokens=1024,
          do_sample=False,
          top_p=None,
          top_k=None,
          temperature=None,
          repetition_penalty=None,
          eos_token_id=text_tokenizer.eos_token_id,
          pad_token_id=text_tokenizer.pad_token_id,
          use_cache=True,
          height=height,
          width=width,
          num_steps=steps,
          seed=seed,
          img_cfg=0,
          txt_cfg=cfg,
      )
    uncond_image = load_blank_image(width, height)
    uncond_prompt = "<image>\nGenerate an image."
    input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
    with torch.inference_mode():
        no_both_cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)
    prompt = "<image>\nDescribe the image by detailing the color, shape, size, texture, quantity, text, and spatial relationships of the objects:" + prompt
    no_txt_cond = None
    input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_inputs(model, text_tokenizer, visual_tokenizer, prompt, uncond_image, width, height)
    with torch.inference_mode():
        cond = model.generate_condition(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, grid_thws=grid_thws, **gen_kwargs)
        cond["vae_pixel_values"] = vae_pixel_values
        images = model.generate_img(cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
    return images


def main():
    args = parse_args()
    model, loading_info = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                torch_dtype=torch.bfloat16,
                                                output_loading_info=True,
                                                trust_remote_code=True
                                                )
    print(f'Loading info of Ovis-U1:\n{loading_info}')

    model = model.eval().to("cuda")
    model = model.to(torch.bfloat16)
    prompt = "a cute cat"
    image = pipe_t2i(model, prompt, args.height, args.width, args.steps, args.txt_cfg)[0]
    image.save("test_t2i.png")


if __name__ == "__main__":
    main()