from PIL import Image, ImageDraw, ImageFont
import textwrap

# Input path (matches the generated artifact path)
bg_path = "/home/techne/.gemini/antigravity/brain/b8b1270f-11ac-475c-9aa8-e0656ce57817/campfire_text_slide_bg_1767769685808.png"
output_path = "/home/techne/aicheckers/public/campfire-text-slide.png"

# Text content
title = "現状の問題"
body_text = """
生成AIの急速な発展により、AIの判別が非常に難しくなっています。

AIそのものの賛否はともかく、生成画像を自作と偽って投稿したり、AI禁止のプラットフォームで不正にマネタイズするような行為は明確に悪と言えるでしょう。

しかし残念ながら、現状はどのプラットフォームも対策が万全とは言い難く、直感によってAI作品を見分けるか、ユーザーからの指摘を受けて対応するしかないのが現状です。

そこでこうした状況に対策を講じるべく、二次元イラストに特化した日本向けのAIチェッカーを開発しました。
"""

# Load image
img = Image.open(bg_path)
draw = ImageDraw.Draw(img)
width, height = img.size

# Fonts (Using Noto Sans CJK JP)
# Adjust sizes relative to image height (assuming 3:2, likely approx 1536x1024 or similar)
# We'll use a dynamic scaling based on height
scale_factor = height / 1024
title_size = int(64 * scale_factor)
body_size = int(32 * scale_factor)

try:
    title_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", title_size)
    body_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", body_size)
    # Fallback to bold if first attempt fails or just regular
except IOError:
    # Try another path if the specific bold one fails, usually NotoSansCJK-Bold.ttc is consistent
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", title_size)
        body_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", body_size)
    except:
        # Fallback to any available font
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

# Layout
margin_x = int(100 * scale_factor)
margin_y = int(100 * scale_factor)
current_y = margin_y

# Draw Title
draw.text((margin_x, current_y), title, font=title_font, fill="#FFFFFF")
current_y += title_size * 2  # Spacing after title

# Draw Body
# We need to wrap text manually or use a library, but raw newlines are honored by PIL for basic multi-line
# However, we want word wrap. Textwrap in python works on characters/width.
# Approximate char width for CJK is full width.
chars_per_line = int((width - (margin_x * 2)) / (body_size)) 

# Filter out empty lines for cleaner processing, then re-add spacing
paragraphs = body_text.strip().split('\n\n')

for p in paragraphs:
    lines = textwrap.wrap(p, width=chars_per_line)
    for line in lines:
        draw.text((margin_x, current_y), line, font=body_font, fill="#E0E0E0")
        current_y += int(body_size * 1.6) # Line height
    current_y += int(body_size * 1.5) # Paragraph spacing

# Save
img.save(output_path)
print(f"Slide saved to {output_path}")
