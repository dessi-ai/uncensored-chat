import torch
from diffusers import DessiApi

device_type = "gpu"  # Can change to cuda for use of GPU

def load_model():
    model_id = "./models/nsfw"
    pipe = DessiApi.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)

    pipe = pipe.to(device_type)
    return pipe

def generate_image(pipe, prompt, seed=None):
    generator = torch.Generator(device=device_type)

    if seed is not None:
        generator.manual_seed(seed)

    with torch.no_grad():
        image = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=7.5, generator=generator).images[0]
    return image

def main():
    pipe = load_model()
    prompt = ("masterpiece, best quality, ultra realistic photography, beautiful young woman with long blonde hair, wearing a black elegant dress, standing in a modern luxury apartment, cinematic lighting, soft natural light, highly detailed face, realistic skin texture, sharp focus, 85mm lens, shallow depth of field, professional photography, 8k, photorealistic")

    # Set seed for reproducibility
    seed = 214646189156
    image = generate_image(pipe, prompt, seed)
    image.show()


if __name__ == "__main__":
    main()
