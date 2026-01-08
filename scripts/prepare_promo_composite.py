from PIL import Image, ImageOps

# Input paths
base_img_path = "/home/techne/.gemini/antigravity/brain/95d2059a-032c-42f4-9dc4-e24583a27e9e/uploaded_image_0_1767840883188.jpg"
heatmap_path = "/home/techne/.gemini/antigravity/brain/95d2059a-032c-42f4-9dc4-e24583a27e9e/uploaded_image_1_1767840883188.png"
output_path = "/home/techne/aicheckers/scripts/promo_composite_input.png"

# Target: 3:2 aspect ratio
target_w, target_h = 1500, 1000

try:
    # Load and resize base image to 3:2
    base = Image.open(base_img_path).convert("RGBA")
    base = ImageOps.fit(base, (target_w, target_h), method=Image.Resampling.LANCZOS)
    
    # Load heatmap image
    heatmap = Image.open(heatmap_path).convert("RGBA")
    
    # Scale heatmap to fit right side (approx 40% width, 70% height)
    hm_target_h = int(target_h * 0.70)
    ratio = hm_target_h / heatmap.height
    hm_new_w = int(heatmap.width * ratio)
    hm_new_h = hm_target_h
    
    heatmap_resized = heatmap.resize((hm_new_w, hm_new_h), Image.Resampling.LANCZOS)
    
    # Position: right side with margin
    x_pos = target_w - hm_new_w - 80
    y_pos = (target_h - hm_new_h) // 2
    
    # Paste heatmap onto base
    base.paste(heatmap_resized, (x_pos, y_pos), heatmap_resized)
    
    base.save(output_path)
    print(f"Composite saved to {output_path}")
    print(f"Size: {base.size}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
