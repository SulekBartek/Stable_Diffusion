import torch
from PIL import Image
from sd_model.predict import generate
import sd_model.util.model_converter as model_converter
from transformers import CLIPTokenizer
from sd_model.model.clip import CLIP
from sd_model.util.load_config import PACKAGE_ROOT
import torchvision.transforms as transforms


def load_clip_model(cfg):

    model_file = PACKAGE_ROOT / cfg.ckpt_path
    state_dict = model_converter.load_from_standard_weights(model_file, cfg.device)

    tokenizer = CLIPTokenizer(
        (PACKAGE_ROOT / cfg.vocab_file_path),
        merges_file=(PACKAGE_ROOT / cfg.merges_file_path),
    )

    clip = CLIP().to(cfg.device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return tokenizer, clip


def preprocess_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(image).unsqueeze(0)


def calculate_clip_score(model, tokenizer, prompt, image, device):

    image = preprocess_image(image).to(device)

    # Tokenize and encode the prompt and image
    text_tokens = tokenizer(
        prompt, return_tensors="pt", truncation=True, padding=True
    ).to(device)
    text_features = model.encode_text(text_tokens.input_ids)
    image_features = model.encode_image(image)

    # Calculate cosine similarity
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = torch.mm(image_features, text_features.t()).item()

    return similarity


def test_clip_score(config):
    device = config["device"]
    tokenizer, model = load_clip_model(config)
    prompt = config["prompt"]

    generated_image = generate(
        config["prompt"],
        config["uncond_prompt"],
        "",
        config["strength"],
        config["do_cfg"],
        config["cfg_scale"],
        config["num_inference_steps"],
        config["seed"],
        config["device"],
        config["idle_device"],
    )

    generated_image = Image.fromarray(generated_image)

    clip_score = calculate_clip_score(model, tokenizer, prompt, generated_image, device)
    print(f"CLIP score: {clip_score}")

    min_clip_score = 0.25
    assert clip_score > min_clip_score
