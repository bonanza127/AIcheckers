
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# Configuration
output_path = "/home/techne/aicheckers/public/campfire-btob-card.png"
texture_path = "/home/techne/.gemini/antigravity/brain/f4be8876-29f9-468e-893b-900810aef98c/black_metal_texture_1767787004129.png"
width, height = 900, 600

# Platinum/Silver Gradient
# Top (White), Middle (Silver), Bottom (White Reflection)
color_stops = [
    (0.0, (255, 255, 255)), # Top
    (0.45, (192, 192, 192)), # Horizon line (darker)
    (0.55, (160, 160, 160)), # Horizon line (darkest)
    (1.0, (240, 240, 245))  # Bottom reflection
]

def create_metallic_gradient(size, stops):
    base = Image.new("RGBA", size, (0,0,0,0))
    draw = ImageDraw.Draw(base)
    
    w, h = size
    for y in range(h):
        ratio = y / h
        # Find segment
        c1 = stops[0][1]
        c2 = stops[-1][1]
        
        for i in range(len(stops) - 1):
            if stops[i][0] <= ratio <= stops[i+1][0]:
                r_start = stops[i][0]
                r_end = stops[i+1][0]
                local_ratio = (ratio - r_start) / (r_end - r_start)
                
                c_start = stops[i][1]
                c_end = stops[i+1][1]
                
                r = int(c_start[0] * (1 - local_ratio) + c_end[0] * local_ratio)
                g = int(c_start[1] * (1 - local_ratio) + c_end[1] * local_ratio)
                b = int(c_start[2] * (1 - local_ratio) + c_end[2] * local_ratio)
                
                draw.line((0, y, w, y), fill=(r,g,b, 255))
                break
    return base

try:
    # 1. Background (Deep Rich Black)
    bg = Image.new("RGBA", (width, height), "#080808")
    
    # Texture Overlay (Subtle)
    try:
        tex = Image.open(texture_path).convert("RGBA")
        tex = tex.resize((width, height), Image.Resampling.LANCZOS)
        tex.putalpha(60) # Slightly more visible for "material" feel
        bg.paste(tex, (0,0), tex)
    except:
        pass

    # Spotlight/Vignette Effect
    # Create a radial gradient from center
    spotlight = Image.new("L", (width, height), 0)
    s_draw = ImageDraw.Draw(spotlight)
    # Draw huge ellipse center
    s_draw.ellipse((width//2 - 400, height//2 - 300, width//2 + 400, height//2 + 300), fill=40)
    spotlight = spotlight.filter(ImageFilter.GaussianBlur(100))
    
    # Composite spotlight (Lighten)
    spot_layer = Image.new("RGBA", (width, height), "#303030")
    spot_layer.putalpha(spotlight)
    bg = Image.alpha_composite(bg, spot_layer)

    draw = ImageDraw.Draw(bg)

    # 2. Font
    font_path = "/home/techne/.local/share/fonts/ShipporiMincho-Regular.ttf"
    try:
        # Reduced font size for shorter text but BtoB is longer than VIP
        font_vip = ImageFont.truetype(font_path, 260) 
    except OSError:
        font_vip = ImageFont.load_default()

    # 3. Text Preparation
    text_main = "BtoB"
    bbox = draw.textbbox((0,0), text_main, font=font_vip)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    
    # Center Position
    tx = (width - tw) // 2
    ty = (height - th) // 2
    
    # Adjust for padding in gradient/shadow images
    pad = 50
    final_w = tw + pad * 2
    final_h = th + pad * 2

    # A. Drop Shadow (Soft & Deep for "Floating" effect)
    shadow_mask = Image.new("L", (final_w, final_h), 0)
    d_sm = ImageDraw.Draw(shadow_mask)
    d_sm.text((pad - bbox[0], pad - bbox[1]), text_main, font=font_vip, fill=255)
    
    shadow_layer = Image.new("RGBA", (final_w, final_h), (0,0,0,0))
    shadow_layer.paste((0,0,0,180), (0,0), shadow_mask) # Dark shadow
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(15))
    
    # Paste shadow onto BG first
    bg.paste(shadow_layer, (tx - pad, ty - pad + 10), shadow_layer) # +10 Y for depth

    # B. Main Text (Metallic Gradient)
    grad_img = create_metallic_gradient((final_w, final_h), color_stops)
    
    # Mask for text
    text_mask = Image.new("L", (final_w, final_h), 0)
    d_tm = ImageDraw.Draw(text_mask)
    d_tm.text((pad - bbox[0], pad - bbox[1]), text_main, font=font_vip, fill=255)
    
    # Apply gradient
    final_text_img = Image.new("RGBA", (final_w, final_h), (0,0,0,0))
    final_text_img.paste(grad_img, (0,0), text_mask)
    
    bg.paste(final_text_img, (tx - pad, ty - pad), final_text_img)

    bg.convert("RGB").save(output_path)
    print(f"Luxury BtoB Card saved to {output_path}")

except Exception as e:
    print(f"Script failed: {e}")
