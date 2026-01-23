
from PIL import Image
import numpy as np
import os

# Create random noise image (should be classified as Real or Low Confidence, definitely not High Quality AI)
# Actually, random noise might be OOD.
# Let's try to make a simple "photo-like" gradient or verify if I can just download one.
# For now, noise is a good "null" hypothesis.

img_data = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
img = Image.fromarray(img_data)
img.save("noise_test.png")
print("Saved noise_test.png")
