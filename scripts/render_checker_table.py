from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

# Input path (Clean Background)
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_text_slide_bg_1767769685808.png"
output_path = "/home/techne/aicheckers/public/campfire-checker-table.png"

# Text Content
title_main = "AIイラストチェッカー"
title_sub = "独自モデル「Moonlight」"
description = """
ViT（Vision Transformer）をベースに、
大量のAI生成画像を学習させた独自モデル「Moonlight」を搭載。
アニメイラスト特有の痕跡を高精度で検出します。
"""

# Table Data (2x2 Grid)
# Title, Percentage, Description, ColorTheme
grid_data = [
    {
        "name": "SDXL", 
        "score": "98.30%", 
        "desc": "ほぼ対策済", 
        "color": "#8b5cf6" # Purple
    },
    {
        "name": "Illustrious", 
        "score": "97.80%", 
        "desc": "派生版含め高頻度で検出", 
        "color": "#8b5cf6"
    },
    {
        "name": "Pony Diffusion", 
        "score": "99.10%", 
        "desc": "現在メインのv6まで\n高頻度で検出", 
        "color": "#8b5cf6"
    },
    {
        "name": "NovelAI", 
        "score": "67.50%", 
        "desc": "やや弱い。\nv4.5対応強化中", 
        "color": "#eab308" # Yellow/Warning
    }
]

# --- LOAD & RESIZE BACKGROUND ---
target_w, target_h = 1500, 1000 # 3:2 Aspect Ratio

try:
    src_img = Image.open(bg_path).convert("RGBA")
    # Resize and crop to fill target
    src_w, src_h = src_img.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h
    
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
        
except FileNotFoundError:
    img = Image.new("RGBA", (target_w, target_h), "#0f0518")

draw = ImageDraw.Draw(img)
width, height = img.size

# --- FONTS ---
scale_factor = height / 1024
try:
    font_path_serif = "/home/techne/.local/share/fonts/ShipporiMincho-Regular.ttf"
    # Basic usage
    title_font = ImageFont.truetype(font_path_serif, int(80 * scale_factor))
    sub_font = ImageFont.truetype(font_path_serif, int(50 * scale_factor))
    desc_font = ImageFont.truetype(font_path_serif, int(32 * scale_factor))
    
    # Fonts for cards (Maybe Sans for tech feel?, sticking to Serif for elegance as requested generally, 
    # but the reference image looks like Sans. Let's try to match the "Business" vibe which used Serif.
    # Actually, readable numbers might be better in Sans or sticking to Serif. Let's stick to Shippori for consistency)
    card_title_font = ImageFont.truetype(font_path_serif, int(40 * scale_factor))
    card_score_font = ImageFont.truetype(font_path_serif, int(40 * scale_factor))
    card_desc_font = ImageFont.truetype(font_path_serif, int(26 * scale_factor))
except:
    title_font = ImageFont.load_default()
    sub_font = ImageFont.load_default()
    desc_font = ImageFont.load_default()
    card_title_font = ImageFont.load_default()
    card_score_font = ImageFont.load_default()
    card_desc_font = ImageFont.load_default()

# --- LAYOUT - HEADER ---
margin_x = int(100 * scale_factor)
current_y = int(80 * scale_factor)

# Title
draw.text((margin_x, current_y), title_main, font=title_font, fill="#FFFFFF")
current_y += int(80 * scale_factor * 1.5)

# Subtitle
draw.text((margin_x, current_y), title_sub, font=sub_font, fill="#d8b4fe")
current_y += int(50 * scale_factor * 1.5)

# Description text
lines = description.strip().split('\n')
for line in lines:
    draw.text((margin_x, current_y), line, font=desc_font, fill="#E0E0E0")
    current_y += int(32 * scale_factor * 1.6)

# --- LAYOUT - GRID ---
current_y += int(40 * scale_factor) # Spacing before grid

grid_start_y = current_y
grid_width = width - (margin_x * 2)
# Calculate card size
# 2 cols, 2 rows
col_gap = int(40 * scale_factor)
row_gap = int(40 * scale_factor)
card_w = (grid_width - col_gap) // 2
card_h = int(180 * scale_factor)

def draw_card(x, y, data):
    # Background
    # Darker box with border
    draw.rounded_rectangle(
        (x, y, x + card_w, y + card_h),
        radius=15,
        fill="#1e1b4b", # Dark indigo
        outline="#4c1d95",
        width=2
    )
    
    # Header area inside card? No, just text layout like standard dashboard card.
    # Title + Score
    padding = int(30 * scale_factor)
    tx = x + padding
    ty = y + padding
    
    # Name
    draw.text((tx, ty), data["name"], font=card_title_font, fill="#FFFFFF")
    
    # Score (Next to name or below? Reference has name.... score)
    # Let's put score nicely colored next to name
    name_bbox = draw.textbbox((tx, ty), data["name"], font=card_title_font)
    name_w = name_bbox[2] - name_bbox[0]
    
    score_x = tx + name_w + 20
    draw.text((score_x, ty), f"({data['score']})", font=card_score_font, fill=data["color"])
    
    # Description
    desc_y = ty + int(40 * scale_factor * 1.5)
    
    # Handle multi-line description
    d_lines = data["desc"].split('\n')
    for dl in d_lines:
        draw.text((tx, desc_y), dl, font=card_desc_font, fill="#94a3b8") # Slate-400 equivalent
        desc_y += int(26 * scale_factor * 1.5)

# Draw cards
for i, item in enumerate(grid_data):
    row = i // 2
    col = i % 2
    
    cx = margin_x + col * (card_w + col_gap)
    cy = grid_start_y + row * (card_h + row_gap)
    
    draw_card(cx, cy, item)

# --- ADD BRANDING/LOGO IF SIMPLE ---
# Maybe a small icon near title if desired, but sticking to clean text is safer.

# Save
img = img.convert("RGB")
img.save(output_path)
print(f"Checker table slide saved to {output_path}")
