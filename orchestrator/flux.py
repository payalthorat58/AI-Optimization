import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="fal-ai",
    api_key="",
)

image = client.text_to_image(
    "Astronaut riding a horse",
    model="black-forest-labs/FLUX.1-dev",
)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Save image
image.save("outputs/generated_image.png")

print("Image saved successfully!")