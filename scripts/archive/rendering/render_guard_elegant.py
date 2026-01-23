from PIL import Image, ImageDraw, ImageFont

# Input path
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_guard_bg_elegant_1767770808777.png"
output_path = "/home/techne/aicheckers/public/campfire-guard-elegant-final.png"

# Text Content
title = "AIイラストガード"
points = [
    "人間には見えにくいノイズで\nAI学習を的確に妨害",
    "NightshadeやGlazeの弱点を\n克服した次世代型AIポイズニング",
    "LoRA学習で効果実証済み"
]

# Load Image
img = Image.open(bg_path)
draw = ImageDraw.Draw(img)
width, height = img.size

# Dynamic Sizes
scale_factor = height / 1024
title_size = int(80 * scale_factor)
point_size = int(36 * scale_factor)
spacing = int(40 * scale_factor)

# Fonts - Using Shippori Mincho for Elegance (as discovered in fc-list previously)
try:
    # Attempt to use the Shippori Mincho found in .local
    font_path = "/home/techne/.local/share/fonts/ShipporiMincho-Regular.ttf"
    title_font = ImageFont.truetype(font_path, title_size)
    point_font = ImageFont.truetype(font_path, point_size)
except:
    # Fallback to Noto Serif if specifically Shippori fails
    try:
        font_path = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
        title_font = ImageFont.truetype(font_path, title_size)
        point_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", point_size)
    except:
        title_font = ImageFont.load_default()
        point_font = ImageFont.load_default()

# Layout
start_x = int(80 * scale_factor)
start_y = int(250 * scale_factor) # Lowered starting position for elegance

# Draw Title
draw.text((start_x, start_y), title, font=title_font, fill="#FFFFFF")
start_y += title_size + int(60 * scale_factor)

# Draw Points with Elegant Lines instead of Bullets
line_length = int(40 * scale_factor)

for point in points:
    # Draw Elegant Line Accent
    line_y = start_y + int(point_size/2)
    draw.line((start_x, line_y, start_x + line_length, line_y), fill="#d8b4fe", width=2)
    
    # Draw Text
    text_x = start_x + line_length + int(20 * scale_factor)
    
    lines = point.split('\n')
    for i, line in enumerate(lines):
        draw.text((text_x, start_y), line, font=point_font, fill="#E0E0E0")
        start_y += int(point_size * 1.6)
    
    start_y += spacing

# Save
img.save(output_path)
print(f"Elegant Guard slide saved to {output_path}")
