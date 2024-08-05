import torch
from sd_model.predict import generate
import torchvision.transforms as transforms

# -------------------------TODO-------------------------


# def generate_adversarial_example(image, config):
#     pass


# def preprocess_image(image):
#     preprocess = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return preprocess(image).unsqueeze(0)


# def test_adversarial_example(config):

#     device = config.device

#     # Generate a clean and adversarial images
#     clean_image = generate(
#                     config["prompt"],
#                     config["uncond_prompt"],
#                     "",
#                     config["strength"],
#                     config["do_cfg"],
#                     config["cfg_scale"],
#                     config["num_inference_steps"],
#                     config["seed"],
#                     config["device"],
#                     config["idle_device"],
#                 )

#     adversarial_image = generate_adversarial_example(clean_image, config)


#     # Preprocess images for comparison
#     clean_image_tensor = preprocess_image(clean_image).to(device)
#     adversarial_image_tensor = preprocess_image(adversarial_image).to(device)

#     # Calculate similarity based on MSE
#     mse = torch.nn.functional.mse_loss(clean_image_tensor, adversarial_image_tensor).item()
#     print(f"MSE between clean and adversarial generated images: {mse}")

#     max_mse_threshold = 0.1  
#     assert mse > max_mse_threshold
