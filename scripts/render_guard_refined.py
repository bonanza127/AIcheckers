from PIL import Image, ImageDraw, ImageFont, ImageFilter

# Paths
# 1. Source for Monitor Frame (The High Quality Template)
monitor_source_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_checker_template_refined_1767767244070.png" 
# Note: Using the 'refined' one might be cleaner, or 'final_stats'. Let's use 'refined' as it had less text clutter near the monitor maybe?
# Actually 'final_stats' had the visual bar chart. 'refined' had just text. 'refined' layout is safer for cropping right side.

# 2. Background Base (Clean Tech BG)
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_text_slide_bg_1767769685808.png"

output_path = "/home/techne/aicheckers/public/campfire-guard-refined.png"

# Text
title = "AIイラストガード"
points = [
    "人間には見えにくいノイズで\nAI学習を的確に妨害",
    "NightshadeやGlazeの弱点を\n克服した次世代型AIポイズニング",
    "LoRA学習で効果実証済み"
]

# Load Images
monitor_img = Image.open(monitor_source_path)
bg_img = Image.open(bg_path)
width, height = bg_img.size

# --- COMPOSITE MONITOR ---
# Assume Monitor is on the right side. We'll crop the right ~55% of the monitor source
# and paste it onto the BG. 
# We need to blend it carefully? Or just a hard paste if the backgrounds match (both dark purple).
# They are from same generation batch prompt style, so likely match. 
# Better: Create a mask gradient?
crop_x = int(width * 0.45)
monitor_crop = monitor_img.crop((crop_x, 0, width, height))

# Paste monitor onto BG
# To ensure smooth transition, let's just paste it.
bg_img.paste(monitor_crop, (crop_x, 0))

# Now we have the Monitor on the Right, and Clean BG on Left (overwriting any previous text from template if it extended this far, but template text was on left).
# Wait, 'monitor_source_path' has text on the left. We cropped the *Right* side, so we kept the Monitor and discarded the text. Perfect.

# --- DRAW SHIELD ICON ---
# Monitor screen area estimation based on 'refined' template
# It's roughly centered in the right half.
screen_cx = crop_x + (width - crop_x) // 2
screen_cy = height // 2
# Fine tune based on visual inspection of previous artifacts: Monitor is massive.
# Let's define the screen bounds roughly to draw the icon inside.
icon_size = int(height * 0.35)

# Shield Drawing Function (Draws onto a separate transparent layer first usually, but direct is fine)
overlay = Image.new("RGBA", bg_img.size, (0,0,0,0))
o_draw = ImageDraw.Draw(overlay)

# Shield Points
sx, sy = screen_cx, screen_cy - int(icon_size * 0.1) # Up a bit
w = int(icon_size * 0.8)
h = int(icon_size * 1.0)
x0 = sx - w//2
y0 = sy - h//2

shield_pts = [
    (x0, y0),
    (sx + w//2, y0),
    (sx + w//2, y0 + h*0.6),
    (sx, y0 + h),
    (x0, y0 + h*0.6)
]

# Glassy Shield Style
# 1. Fill (Semi-transparent Gradient-ish)
o_draw.polygon(shield_pts, fill=(139, 92, 246, 100)) # Light Purple Translucent

# 2. Stroke (Glowing)
o_draw.line(shield_pts + [shield_pts[0]], fill=(255, 255, 255, 200), width=4)

# 3. Inner details - High tech lines
# Horizontal scan line
y_scan = sy
o_draw.line((x0 + w*0.1, y_scan, sx + w*0.4, y_scan), fill=(255, 255, 255, 150), width=2)
# Vertical line
o_draw.line((sx, y0 + h*0.1, sx, y0 + h*0.9), fill=(255, 255, 255, 100), width=2)

# 4. "AI" Text inside? Use "LOCK" symbol for Guard? Or user said "Simple Shield Icon".
# Let's draw a simple Lock cut-out or shape center
lock_w = w * 0.3
lock_h = h * 0.3
lx = sx - lock_w/2
ly = sy - lock_h/4 # slightly up

# Lock body
o_draw.rectangle((lx, ly + lock_h*0.4, lx + lock_w, ly + lock_h), fill=(255,255,255,200))
# Lock shackle
o_draw.arc((lx + lock_w*0.1, ly, lx + lock_w*0.9, ly + lock_h*0.8), 180, 0, fill=(255,255,255,200), width=3)


# Paste Overlay
bg_img = Image.alpha_composite(bg_img.convert("RGBA"), overlay)


# --- TEXT RENDERING (Left Side) ---
draw = ImageDraw.Draw(bg_img)
scale_factor = height / 1024

# Layout
left_margin = int(80 * scale_factor)
start_y = int(220 * scale_factor)

# Fonts
try:
    font_path = "/home/techne/.local/share/fonts/ShipporiMincho-Regular.ttf"
    title_font = ImageFont.truetype(font_path, int(90 * scale_factor)) # Larger title
    point_font = ImageFont.truetype(font_path, int(38 * scale_factor))
except:
    title_font = ImageFont.load_default()
    point_font = ImageFont.load_default()

# Title
draw.text((left_margin, start_y), title, font=title_font, fill="#FFFFFF")
start_y += int(140 * scale_factor)

# Points with nice separators
for point in points:
    # Small vertical bar accent
    bar_h = int(38 * scale_factor * 1.5)
    # Check lines
    lines = point.split('\n')
    total_text_h = len(lines) * int(38*scale_factor*1.6)
    
    # Draw accent bar
    draw.rectangle((left_margin, start_y + 5, left_margin + 4, start_y + total_text_h - 5), fill="#d8b4fe")
    
    text_x = left_margin + 25
    for line in lines:
        draw.text((text_x, start_y), line, font=point_font, fill="#f3e8ff")
        start_y += int(38 * scale_factor * 1.6)
    
    start_y += int(40 * scale_factor) # Spacing


# Save
bg_img.save(output_path)
print(f"Refined Guard slide saved to {output_path}")
