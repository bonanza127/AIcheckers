from PIL import Image, ImageDraw, ImageFont, ImageFilter
import math

# Input path (Clean Background)
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_text_slide_bg_1767769685808.png"
output_path = "/home/techne/aicheckers/public/campfire-guard-business.png"

# Text Content
title = "AIイラストガード"
subtitle = "クリエイターの権利を守る盾"
points = [
    "人間には見えにくいノイズで\nAI学習を的確に妨害",
    "NightshadeやGlazeの弱点を\n克服した次世代型AIポイズニング",
    "LoRA学習で効果実証済み"
]

# Load Image
target_w, target_h = 1500, 1000 # 3:2 Aspect Ratio

try:
    src_img = Image.open(bg_path).convert("RGBA")
    # Resize and crop to fill target
    src_w, src_h = src_img.size
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h
    
    if src_ratio > target_ratio:
        # Source is wider, scale by height
        scale = target_h / src_h
        new_w = int(src_w * scale)
        new_h = target_h
        src_img = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Center crop width
        left = (new_w - target_w) // 2
        img = src_img.crop((left, 0, left + target_w, new_h))
    else:
        # Source is taller (or equal), scale by width
        scale = target_w / src_w
        new_w = target_w
        new_h = int(src_h * scale)
        src_img = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Center crop height
        top = (new_h - target_h) // 2
        img = src_img.crop((0, top, new_w, top + target_h))
        
except FileNotFoundError:
    # Create a dummy background if file not found (fallback)
    img = Image.new("RGBA", (target_w, target_h), "#0f0518")

draw = ImageDraw.Draw(img)
width, height = img.size

# --- LAYOUT CONSTANTS ---
scale_factor = height / 1024
left_margin = int(100 * scale_factor)
text_start_y = int(200 * scale_factor)

# Fonts
try:
    font_path_serif = "/home/techne/.local/share/fonts/ShipporiMincho-Regular.ttf"
    title_font = ImageFont.truetype(font_path_serif, int(90 * scale_factor))
    subtitle_font = ImageFont.truetype(font_path_serif, int(40 * scale_factor))
    point_font = ImageFont.truetype(font_path_serif, int(38 * scale_factor))
except:
    # Fallback fonts
    title_font = ImageFont.load_default()
    subtitle_font = ImageFont.load_default()
    point_font = ImageFont.load_default()

# --- DRAW TEXT (Left Side) ---
# Title
draw.text((left_margin, text_start_y), title, font=title_font, fill="#FFFFFF")

# Subtitle (Elegant addition for business feel)
current_y = text_start_y + int(90 * scale_factor) + int(20 * scale_factor)
draw.text((left_margin, current_y), subtitle, font=subtitle_font, fill="#d8b4fe") # Light purple accent

current_y += int(40 * scale_factor * 2) + int(40 * scale_factor)

# Points with styled bullets
bullet_size = int(10 * scale_factor)
point_spacing = int(50 * scale_factor)

for point in points:
    # Diamond bullet
    bullet_y = current_y + int(38 * scale_factor/2) - bullet_size//2 + 5
    draw.polygon([
        (left_margin, bullet_y),
        (left_margin + bullet_size, bullet_y - bullet_size),
        (left_margin + bullet_size * 2, bullet_y),
        (left_margin + bullet_size, bullet_y + bullet_size)
    ], fill="#8b5cf6") 

    # Text
    text_x = left_margin + bullet_size * 3 + int(10 * scale_factor)
    lines = point.split('\n')
    for i, line in enumerate(lines):
        line_color = "#E0E0E0"
        # Highlight specific keywords if needed, but for now simple elegant white/grey
        draw.text((text_x, current_y), line, font=point_font, fill=line_color)
        current_y += int(38 * scale_factor * 1.5)
    current_y += point_spacing





# --- DRAW VISUAL (Right Side - Lucide Shield Icon) ---
# User requested the 'Shield' icon from the FAQ (Lucide React Shield).
# We have rendered this to 'lucide_shield.png'.

shield_path = "/home/techne/aicheckers/scripts/lucide_shield.png"

# Target area for the graphic
center_x = int(width * 0.75)
center_y = int(height * 0.5)
graphic_max_h = int(height * 0.60)
graphic_max_w = int(width * 0.45)

try:
    shield_img = Image.open(shield_path).convert("RGBA")
    
    # Resize to fit within constraints
    s_w, s_h = shield_img.size
    ratio = min(graphic_max_w / s_w, graphic_max_h / s_h)
    new_s_w = int(s_w * ratio)
    new_s_h = int(s_h * ratio)
    
    shield_img = shield_img.resize((new_s_w, new_s_h), Image.Resampling.LANCZOS)
    
    # Add a subtle glow behind the shield lines
    # Since it's a line art, we can add a glow by blurring a copy of the alpha channel or the image itself
    
    glow_size = 20
    glow_layer = Image.new("RGBA", (new_s_w + glow_size*2, new_s_h + glow_size*2), (0,0,0,0))
    
    # Paste shield into glow layer center
    glow_layer.paste(shield_img, (glow_size, glow_size), shield_img)
    
    # Create glow color version
    # Extract alpha
    alpha = glow_layer.split()[3]
    # Create solid color
    glow_color = Image.new("RGBA", glow_layer.size, "#8b5cf6")
    glow_color.putalpha(alpha)
    
    # Blur
    glow_blurred = glow_color.filter(ImageFilter.GaussianBlur(10))
    
    # Composite glow, then shield
    # Calculate position on main image
    final_x = center_x - new_s_w // 2
    final_y = center_y - new_s_h // 2
    
    # Draw a larger soft radial background to make it pop against the dark BG
    rad_size = int(new_s_w * 1.5)
    rad_layer = Image.new("RGBA", (width, height), (0,0,0,0))
    r_draw = ImageDraw.Draw(rad_layer)
    for r in range(rad_size // 2, 0, -5):
         alpha_rad = int(30 * (1 - r/(rad_size//2)))
         r_draw.ellipse((center_x - r, center_y - r, center_x + r, center_y + r), fill=(139, 92, 246, alpha_rad))
    img = Image.alpha_composite(img, rad_layer)

    # Paste Shield (and its immediate glow)
    # Since we built 'glow_blurred' relative to the shield, let's paste it
    # Adjust paste coordinates for the padding
    img.paste(glow_blurred, (final_x - glow_size, final_y - glow_size), glow_blurred)
    img.paste(shield_img, (final_x, final_y), shield_img)
    
except Exception as e:
    print(f"Error loading shield: {e}")
    # Fallback
    draw.ellipse((center_x - 100, center_y - 100, center_x + 100, center_y + 100), fill="#6d28d9")



# --- SAVE ---
img = img.convert("RGB")
img.save(output_path)
print(f"Business Guard slide saved to {output_path}")
