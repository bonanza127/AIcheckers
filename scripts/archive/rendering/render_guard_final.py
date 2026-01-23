from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Input paths
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_text_slide_bg_1767769685808.png"
bottom_image_path = "/home/techne/.gemini/antigravity/brain/f4be8876-29f9-468e-893b-900810aef98c/uploaded_image_2_1767784938732.png"
shield_icon_path = "/home/techne/aicheckers/scripts/lucide_shield.png"
output_path = "/home/techne/aicheckers/public/campfire-guard-final.png"

# Text Content
title_main = "AIイラストガード"
title_sub = "独自モデル「Moonknight」"
# Summarized bullet points into a clean description block similar to Checker slide
description_lines = [
    "人間に知覚されにくい微細なノイズでAIの認識を妨害。",
    "NightshadeやGlazeの弱点を克服するため、\nいくつかの新技術をかけあわせた次世代型AIポイズニング。",
    "特に少数枚で絵柄を模倣するLoRAに対して防御効果を発揮します。"
]

# --- LOAD & RESIZE BACKGROUND (3:2) ---
target_w, target_h = 1500, 1000

try:
    src_img = Image.open(bg_path).convert("RGBA")
    src_w, src_h = src_img.size
    target_ratio = target_w / target_h
    src_ratio = src_w / src_h
    
    if src_ratio > target_ratio:
        scale = target_h / src_h
        new_w = int(src_w * scale)
        new_h = target_h
        src_img = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        img = src_img.crop((left, 0, left + target_w, new_h))
    else:
        scale = target_w / src_w
        new_w = target_w
        new_h = int(src_h * scale)
        src_img = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        top = (new_h - target_h) // 2
        img = src_img.crop((0, top, new_w, top + target_h))
except:
    img = Image.new("RGBA", (target_w, target_h), "#0f0518")

draw = ImageDraw.Draw(img)
width, height = img.size
scale_factor = height / 1024

# --- FONTS ---
try:
    font_path = "/home/techne/.local/share/fonts/ShipporiMincho-Regular.ttf"
    title_font = ImageFont.truetype(font_path, int(80 * scale_factor))
    sub_font = ImageFont.truetype(font_path, int(50 * scale_factor))
    desc_font = ImageFont.truetype(font_path, int(32 * scale_factor))
except:
    title_font = ImageFont.load_default()
    sub_font = ImageFont.load_default()
    desc_font = ImageFont.load_default()

# --- LAYOUT - HEADER & TEXT ---
margin_x = int(100 * scale_factor)
current_y = int(80 * scale_factor)

# Title
draw.text((margin_x, current_y), title_main, font=title_font, fill="#FFFFFF")
current_y += int(80 * scale_factor * 1.4)

# Subtitle line
draw.text((margin_x, current_y), title_sub, font=sub_font, fill="#d8b4fe")


current_y += int(50 * scale_factor * 1.5)

# Description Blocks
for block in description_lines:
    for line in block.split('\n'):
        draw.text((margin_x, current_y), line, font=desc_font, fill="#E0E0E0")
        current_y += int(32 * scale_factor * 1.6)
    # Add a little extra spacing between main bullet points if needed, or keep uniform
    current_y += int(10 * scale_factor)

# --- LAYOUT - BOTTOM IMAGE ---
diagram_margin_top = int(40 * scale_factor)
current_y += diagram_margin_top

# Available height
remaining_h = height - current_y - int(50 * scale_factor)
# Use max 90% width for this one as it's a wide comparison
max_img_w = int(width * 0.9)

try:
    bottom_img = Image.open(bottom_image_path).convert("RGBA")
    b_w, b_h = bottom_img.size
    
    # Scale to fit
    ratio = min(max_img_w / b_w, remaining_h / b_h)
    new_b_w = int(b_w * ratio)
    new_b_h = int(b_h * ratio)
    
    bottom_img = bottom_img.resize((new_b_w, new_b_h), Image.Resampling.LANCZOS)
    
    # Glow/Shadow container
    # Since it's a screenshot, maybe a simple border
    border_w = 2
    framed_img = Image.new("RGBA", (new_b_w + border_w*2, new_b_h + border_w*2), "#2e1065")
    framed_img.paste(bottom_img, (border_w, border_w))
    
    paste_x = (width - framed_img.width) // 2
    paste_y = current_y + (remaining_h - framed_img.height) // 2 # Center vertically in remaining space
    
    img.paste(framed_img, (paste_x, paste_y))
    
except Exception as e:
    print(f"Error loading bottom image: {e}")

img = img.convert("RGB")
img.save(output_path)
print(f"Final Guard slide saved to {output_path}")
