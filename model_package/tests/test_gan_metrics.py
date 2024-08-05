import torch
from torchvision import transforms
from torch_fidelity import calculate_metrics
from sd_model.predict import generate


def preprocess_images(images):
    """Preprocess images for FID calculation."""
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return torch.stack([preprocess(image) for image in images])



def test_fid_and_ic(real_images, input_text_to_image):
    """Test the FID and inception score between a single generated image and a set of real images."""

    # modify prompt based on real-world dataset
    generated_image = generate(*input_text_to_image)

    real_images = preprocess_images(real_images)
    generated_image = preprocess_images([generated_image])

    metrics = calculate_metrics(
        input1=real_images,
        input2=generated_image,
        isc=True, 
        fid=True, 
        verbose=True
    )

    assert metrics["frechet_inception_distance"] < 100
    assert metrics["inception_score_mean"] > 10
    assert metrics["inception_score_std"] < 0.1