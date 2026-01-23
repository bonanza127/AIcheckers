from PIL import Image, ImageDraw, ImageFont

# Input path
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_guard_bg_clean_1767770421406.png"
output_path = "/home/techne/aicheckers/public/campfire-guard-final.png"

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
spacing = int(30 * scale_factor)

# Fonts
try:
    title_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", title_size)
    point_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", point_size)
    point_bold_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", point_size)
except:
    title_font = ImageFont.load_default()
    point_font = ImageFont.load_default()
    point_bold_font = ImageFont.load_default()

# Layout
start_x = int(80 * scale_factor)
start_y = int(150 * scale_factor)

# Draw Title
draw.text((start_x, start_y), title, font=title_font, fill="#FFFFFF")
start_y += title_size + int(60 * scale_factor) # Space after title

# Draw Points with Stylized Bullets
bullet_radius = int(8 * scale_factor)

for point in points:
    # Draw Bullet (Gold/Yellow style to match premium feel)
    bullet_y = start_y + int(point_size/2)
    draw.ellipse(
        (start_x, bullet_y - bullet_radius, start_x + bullet_radius*2, bullet_y + bullet_radius),
        fill="#FFD700"
    )
    
    # Draw Text
    text_x = start_x + int(40 * scale_factor)
    # Check if lines need splitting (already split in list for control)
    lines = point.split('\n')
    for i, line in enumerate(lines):
        # Use bold for key terms could be nice, but keep simple for now
        draw.text((text_x, start_y), line, font=point_font, fill="#E0E0E0")
        start_y += int(point_size * 1.5)
    
    start_y += spacing

# Save
img.save(output_path)
print(f"Guard slide saved to {output_path}")
