from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Input path (Clean Background)
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_text_slide_bg_1767769685808.png"
vit_diagram_path = "/home/techne/.gemini/antigravity/brain/f4be8876-29f9-468e-893b-900810aef98c/uploaded_image_1767783672037.png"
output_path = "/home/techne/aicheckers/public/campfire-checker-vit.png"

# Text Content
title_main = "AIイラストチェッカー"
title_sub = "独自モデル「Moonlight」"
description_1 = "AIの生成画像を大量に学習し、\nアニメイラスト用にファインチューニング。"
description_2 = "さらにパッチ統計とアテンションマップを実装することにより、\nViTが何を根拠にAI判定を出したのか、可視化できるようになりました。"

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
    desc_font = ImageFont.truetype(font_path, int(36 * scale_factor))
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

# Subtitle
draw.text((margin_x, current_y), title_sub, font=sub_font, fill="#d8b4fe")
current_y += int(50 * scale_factor * 1.5)

# Text Block 1
for line in description_1.split('\n'):
    draw.text((margin_x, current_y), line, font=desc_font, fill="#E0E0E0")
    current_y += int(36 * scale_factor * 1.6)

current_y += int(20 * scale_factor)

# Text Block 2 (Attention map info)
for line in description_2.split('\n'):
    draw.text((margin_x, current_y), line, font=desc_font, fill="#E0E0E0")
    current_y += int(36 * scale_factor * 1.6)

# --- LAYOUT - DIAGRAM ---
# Position diagram at the bottom, centered horizontally
diagram_margin_top = int(40 * scale_factor)
current_y += diagram_margin_top

# Available height
remaining_h = height - current_y - int(50 * scale_factor)
# Use max 80% width
max_diag_w = int(width * 0.8)

try:
    diag_img = Image.open(vit_diagram_path).convert("RGBA")
    d_w, d_h = diag_img.size
    
    # Scale to fit
    ratio = min(max_diag_w / d_w, remaining_h / d_h)
    new_d_w = int(d_w * ratio)
    new_d_h = int(d_h * ratio)
    
    diag_img = diag_img.resize((new_d_w, new_d_h), Image.Resampling.LANCZOS)
    
    # Draw simple frame/glow for the diagram
    bg_pad = 10
    diag_bg_x = width // 2
    diag_bg_y = current_y + new_d_h // 2
    
    # Glow behind diagram
    glow_layer = Image.new("RGBA", (width, height), (0,0,0,0))
    g_draw = ImageDraw.Draw(glow_layer)
    # Elliptical glow
    g_w_rad = int(new_d_w * 0.6)
    g_h_rad = int(new_d_h * 0.6)
    for r in range(10, 0, -1):
        alpha = int(20 * r/10)
        # Larger blur rectangle? no just simple radial-ish
        g_draw.ellipse(
            (diag_bg_x - g_w_rad - r*10, diag_bg_y - g_h_rad - r*10, 
             diag_bg_x + g_w_rad + r*10, diag_bg_y + g_h_rad + r*10),
            fill=(76, 29, 149, alpha)
        )
    img = Image.alpha_composite(img, glow_layer)
    
    # Paste Diagram
    paste_x = (width - new_d_w) // 2
    paste_y = current_y
    
    img.paste(diag_img, (paste_x, paste_y), diag_img)
    
except Exception as e:
    print(f"Error loading diagram: {e}")

img = img.convert("RGB")
img.save(output_path)
print(f"ViT explanation slide saved to {output_path}")
