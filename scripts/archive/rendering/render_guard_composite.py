from PIL import Image, ImageDraw, ImageFont

# Input path (Clean Background)
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_text_slide_bg_1767769685808.png"
output_path = "/home/techne/aicheckers/public/campfire-guard-composite.png"

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

# --- LAYOUT CONSTANTS ---
scale_factor = height / 1024
left_margin = int(80 * scale_factor)
text_start_y = int(250 * scale_factor)
# Window Layout
window_x = int(width * 0.55)
window_y = int(height * 0.25)
window_w = int(width * 0.40)
window_h = int(height * 0.50)

# Fonts
try:
    font_path = "/home/techne/.local/share/fonts/ShipporiMincho-Regular.ttf"
    title_font = ImageFont.truetype(font_path, int(80 * scale_factor))
    point_font = ImageFont.truetype(font_path, int(36 * scale_factor))
except:
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc", int(80 * scale_factor))
        point_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", int(36 * scale_factor))
    except:
        title_font = ImageFont.load_default()
        point_font = ImageFont.load_default()


# --- DRAW TEXT (Left Side) ---
draw.text((left_margin, text_start_y), title, font=title_font, fill="#FFFFFF")
current_y = text_start_y + int(80 * scale_factor) + int(60 * scale_factor)

line_length = int(40 * scale_factor)
spacing = int(40 * scale_factor)

for point in points:
    # Decorative Line
    line_y = current_y + int(36 * scale_factor/2)
    draw.line((left_margin, line_y, left_margin + line_length, line_y), fill="#d8b4fe", width=2)
    
    # Text
    text_x = left_margin + line_length + int(20 * scale_factor)
    lines = point.split('\n')
    for line in lines:
        draw.text((text_x, current_y), line, font=point_font, fill="#E0E0E0")
        current_y += int(36 * scale_factor * 1.6)
    current_y += spacing


# --- DRAW VISUAL (Right Side - Monitor + Shield) ---

# 1. Monitor Frame
border_color = "#4c1d95" # Dark purple border
screen_bg = "#0f0518" # Very dark purple/black
frame_thickness = int(10 * scale_factor)

# Draw outer frame (Shadow/Bevel effect)
draw.rounded_rectangle(
    (window_x - frame_thickness, window_y - frame_thickness, 
     window_x + window_w + frame_thickness, window_y + window_h + frame_thickness),
    radius=20, fill="#2e1065", outline="#6d28d9", width=2
)
# Draw Screen Area
draw.rectangle(
    (window_x, window_y, window_x + window_w, window_y + window_h),
    fill=screen_bg, outline="#5b21b6", width=2
)

# 2. Simple Shield Icon
# Center of the screen
cx = window_x + window_w // 2
cy = window_y + window_h // 2
shield_w = int(window_w * 0.4)
shield_h = int(window_h * 0.5)

# Points for shield shape
# Top Left, Top Right, Bottom Tip
sx = cx - shield_w // 2
sy = cy - shield_h // 2

# Shield Polygon: (x, y), (x+w, y), (x+w, y+h*0.6), (cx, y+h), (x, y+h*0.6)
points_shield = [
    (sx, sy), 
    (sx + shield_w, sy), 
    (sx + shield_w, sy + int(shield_h * 0.6)), 
    (cx, sy + shield_h), 
    (sx, sy + int(shield_h * 0.6))
]

# Draw Shield with Gradient-like fill (Split in half for simple effect)
# Left half
draw.polygon([
    (sx, sy), (cx, sy), (cx, sy + shield_h), (sx, sy + int(shield_h * 0.6))
], fill="#8b5cf6") # Light purple

# Right half
draw.polygon([
    (cx, sy), (sx + shield_w, sy), (sx + shield_w, sy + int(shield_h * 0.6)), (cx, sy + shield_h)
], fill="#6d28d9") # Darker purple for depth

# White outline
draw.line(points_shield + [(sx, sy)], fill="#c4b5fd", width=3)

# 3. Lock Icon or "AI" text inside Shield?
# Let's draw "AI" inside
ai_size = int(shield_h * 0.3)
try:
    ai_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", ai_size)
except:
    ai_font = ImageFont.load_default()

text_bbox = draw.textbbox((0, 0), "AI", font=ai_font)
text_w = text_bbox[2] - text_bbox[0]
text_h = text_bbox[3] - text_bbox[1]
draw.text((cx - text_w // 2, cy - text_h // 2 - int(shield_h*0.1)), "AI", font=ai_font, fill="#FFFFFF")


# Save
img.save(output_path)
print(f"Composite Guard slide saved to {output_path}")
