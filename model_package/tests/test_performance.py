import time
from sd_model.predict import generate


def test_performance(config):
    """Test the model performance in terms of iterations per second."""

    # Generate a single image with 2 steps to warm up and avoid initial overhead
    config["num_inference_steps"] = 2
    generate(
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

    # Measure the time for generating a batch of images
    batch_size = 10
    config["num_inference_steps"] = 25
    start_time = time.time()

    for _ in range(batch_size):
        generate(
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

    end_time = time.time()

    total_time = end_time - start_time
    iterations_per_second = batch_size / total_time
    iterations_per_minute = iterations_per_second * 60

    print(f"Device used: {config['device']}")
    print(f"Total time for {batch_size} images: {total_time:.2f} seconds")
    print(f"Iterations per second: {iterations_per_minute:.2f}")

    min_iterations_per_minute = 0.1

    assert iterations_per_minute > min_iterations_per_minute