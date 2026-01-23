from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

# Paths
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_text_slide_bg_1767769685808.png"
heatmap_path = "/home/techne/.gemini/antigravity/brain/f4be8876-29f9-468e-893b-900810aef98c/uploaded_image_1767783672037.png"
output_path = "/home/techne/aicheckers/public/checker_promo_refined.png"

# Config
width, height = 1500, 1000

def create_gradient_icon(size, color_start, color_end):
    img = Image.new("RGBA", size, (0,0,0,0))
    draw = ImageDraw.Draw(img)
    # Simple rounded rect
    draw.rounded_rectangle((0,0, size[0], size[1]), radius=size[0]//4, fill=color_start)
    return img

try:
    # 1. Background
    try:
        bg = Image.open(bg_path).convert("RGBA")
        bg = bg.resize((width, height), Image.Resampling.LANCZOS) # Assuming aspect ratio fits or simple stretch for abstract bg
    except:
        bg = Image.new("RGBA", (width, height), "#0f0518")

    draw = ImageDraw.Draw(bg)
    scale = height / 1000

    # Fonts
    font_path = "/home/techne/.local/share/fonts/ShipporiMincho-Regular.ttf"
    try:
        title_font = ImageFont.truetype(font_path, int(130 * scale))
        label_font = ImageFont.truetype(font_path, int(36 * scale))
        value_font = ImageFont.truetype(font_path, int(80 * scale))
        unit_font = ImageFont.truetype(font_path, int(50 * scale))
        desc_font = ImageFont.truetype(font_path, int(32 * scale))
        note_font = ImageFont.truetype(font_path, int(24 * scale))
        banner_font = ImageFont.truetype(font_path, int(30 * scale))
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default() 
        # ... fallback logic

    # Layout Config
    left_margin = int(100 * scale)
    top_margin = int(150 * scale)
    
    # --- LEFT COLUMN ---
    
    # 1. Main Title
    # "AIイラスト\nチェッカー"
    # To match image, it might be 2 lines or 1. Image shows 2 lines? 
    # User image: "AIイラスト" (top) "チェッカー" (bottom) - BIG
    
    current_y = top_margin
    draw.text((left_margin, current_y), "AIイラスト", font=title_font, fill="#fef3c7") # Light cream
    current_y += int(130 * scale * 1.1)
    draw.text((left_margin, current_y), "チェッカー", font=title_font, fill="#fef3c7")
    current_y += int(130 * scale * 1.2)

    # 2. Stats Row
    # We have two stats side by side or stacked? 
    # Image shows "検知精度" and "学習データ" side by side (approx).
    
    stats_y = current_y
    col1_x = left_margin
    col2_x = left_margin + int(350 * scale)
    
    # Stat 1: Accuracy
    # Icon (Simple Circle with bar?)
    # Just text for now or draw simple icon
    draw.ellipse((col1_x, stats_y, col1_x + 30, stats_y + 30), fill="#8b5cf6") # Purple dot icon
    draw.text((col1_x + 40, stats_y), "検知精度", font=label_font, fill="#FFFFFF")
    
    # Value
    val_y = stats_y + int(40 * scale)
    # "98.35" (big) "%" (small) "*" (small/superscript)
    val_text = "98.35"
    unit_text = "%"
    
    draw.text((col1_x, val_y), val_text, font=value_font, fill="#fbbf24") # Amber/Gold
    w_val = draw.textlength(val_text, font=value_font)
    draw.text((col1_x + w_val, val_y + 20), unit_text, font=unit_font, fill="#fbbf24")
    # Asterisk
    w_unit = draw.textlength(unit_text, font=unit_font)
    draw.text((col1_x + w_val + w_unit + 5, val_y + 10), "※", font=note_font, fill="#9ca3af") # Grey asterisk

    # Stat 2: Training Data
    draw.ellipse((col2_x, stats_y, col2_x + 30, stats_y + 30), fill="#8b5cf6") # Purple dot
    draw.text((col2_x + 40, stats_y), "学習データ", font=label_font, fill="#FFFFFF")
    
    # Value
    # "10万枚以上"
    val2_text = "10万枚"
    unit2_text = "以上"
    draw.text((col2_x, val_y), val2_text, font=value_font, fill="#fbbf24")
    w_val2 = draw.textlength(val2_text, font=value_font)
    draw.text((col2_x + w_val2, val_y + 20), unit2_text, font=unit_font, fill="#fbbf24")

    current_y = val_y + int(100 * scale)

    # 3. Model List
    models_text = "SDXL / Illustrious / Pony / NovelAI\nなど主流モデルを網羅"
    draw.text((left_margin, current_y), models_text, font=desc_font, fill="#e2e8f0")
    
    # 4. Disclaimer Note (New request)
    current_y += int(32 * scale * 2.5) # Spacing
    note_text = "※学習データ一万枚で検証、LoRAなしの場合"
    draw.text((left_margin, current_y), note_text, font=note_font, fill="#94a3b8") # Slate 400

    # --- RIGHT SIDE: Heatmap ---
    # Load Image
    try:
        heatmap = Image.open(heatmap_path).convert("RGBA")
        # Resize to fit right side, e.g. 45% of width
        target_h_img = int(height * 0.7)
        h_ratio = target_h_img / heatmap.height
        new_h_w = int(heatmap.width * h_ratio)
        heatmap = heatmap.resize((new_h_w, target_h_img), Image.Resampling.LANCZOS)
        
        # Rounded corners for image
        mask = Image.new("L", (new_h_w, target_h_img), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.rounded_rectangle((0,0, new_h_w, target_h_img), radius=20, fill=255)
        
        # Position
        img_x = width - new_h_w - int(100 * scale)
        img_y = (height - target_h_img) // 2
        
        # Glow Effect
        glow = Image.new("RGBA", (width, height), (0,0,0,0))
        g_draw = ImageDraw.Draw(glow)
        # Simple purple glow behind
        g_draw.ellipse((img_x - 30, img_y + target_h_img//2 - 100, img_x + new_h_w + 30, img_y + target_h_img//2 + 100), fill=(139, 92, 246, 80))
        glow = glow.filter(ImageFilter.GaussianBlur(40))
        bg = Image.alpha_composite(bg, glow)
        
        # Paste Image
        bg.paste(heatmap, (img_x, img_y), mask)
        
        # --- BANNER/LABEL ---
        # "Elegant" style
        # "ヒートマップで判定箇所を可視化"
        # Floating pill at bottom of image
        
        banner_text = "ヒートマップで判定箇所を可視化"
        bbox = draw.textbbox((0,0), banner_text, font=banner_font)
        bw = bbox[2] - bbox[0] + int(60 * scale) # Padding
        bh = bbox[3] - bbox[1] + int(30 * scale)
        
        bx = img_x + (new_h_w - bw) // 2
        by = img_y + target_h_img - bh - int(30 * scale)
        
        # Refined Banner: Dark Glass
        # Draw semi-transparent rounded rect
        banner_layer = Image.new("RGBA", (width, height), (0,0,0,0))
        banner_draw = ImageDraw.Draw(banner_layer)
        
        # Shadow
        banner_draw.rounded_rectangle((bx+2, by+4, bx+bw+2, by+bh+4), radius=bh//2, fill=(0,0,0,100))
        # Main body
        banner_draw.rounded_rectangle((bx, by, bx+bw, by+bh), radius=bh//2, fill=(15, 23, 42, 220)) # Dark slate high opacity
        # Stroke
        banner_draw.rounded_rectangle((bx, by, bx+bw, by+bh), radius=bh//2, outline=(139, 92, 246, 180), width=1)
        
        # Composite banner
        bg = Image.alpha_composite(bg, banner_layer)
        
        # Draw text on top
        txt_x = bx + (bw - (bbox[2] - bbox[0])) // 2
        txt_y = by + (bh - (bbox[3] - bbox[1])) // 2 - 2 # Optical adjustment
        
        # Create a new draw object for final text on top of everything
        draw_final = ImageDraw.Draw(bg)
        draw_final.text((txt_x, txt_y), banner_text, font=banner_font, fill="#f8fafc") # Slate 50

    except Exception as e:
        print(f"Error drawing heatmap: {e}")

    # Save
    bg.save(output_path)
    print(f"Refined Promo Image saved to {output_path}")

except Exception as e:
    print(f"Script failed: {e}")
