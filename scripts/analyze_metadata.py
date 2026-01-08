
from PIL import Image
from PIL.ExifTags import TAGS
import sys

def analyze_image(path):
    print(f"\n--- Analyzing: {path} ---")
    try:
        img = Image.open(path)
        print(f"Format: {img.format}")
        print(f"Mode: {img.mode}")
        print(f"Size: {img.size}")
        
        # 1. Check img.info (Common for PNG "parameters" and sometimes JPEG "Comment")
        if img.info:
            print("--- Image Info ---")
            for k, v in img.info.items():
                # Truncate long values for display
                v_str = str(v)
                if len(v_str) > 500:
                    v_str = v_str[:500] + "..."
                print(f"{k}: {v_str}")
        else:
            print("No Image Info found.")

        # 2. Check Exif Data
        exif_data = img.getexif()
        if exif_data:
            print("--- Exif Data ---")
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                v_str = str(value)
                if len(v_str) > 100:
                    v_str = v_str[:100] + "..."
                print(f"{tag_name}: {v_str}")
        else:
            print("No Exif Data found.")

    except Exception as e:
        print(f"Error analyzing image: {e}")

if __name__ == "__main__":
    paths = [
        "/home/techne/.gemini/antigravity/brain/f4be8876-29f9-468e-893b-900810aef98c/uploaded_image_0_1767792025871.jpg",
        "/home/techne/.gemini/antigravity/brain/f4be8876-29f9-468e-893b-900810aef98c/uploaded_image_1_1767792025871.jpg"
    ]
    for p in paths:
        analyze_image(p)
