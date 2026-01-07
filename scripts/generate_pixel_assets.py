
import os
from PIL import Image, ImageDraw

def grid_to_svg(grid, palette, pixel_size=10):
    height = len(grid)
    width = len(grid[0])
    
    svg_content = f'<svg width="{width * pixel_size}" height="{height * pixel_size}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">\n'
    
    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char in palette and palette[char] is not None:
                color_info = palette[char]
                if isinstance(color_info, tuple):
                    color, opacity = color_info
                    svg_content += f'  <rect x="{x}" y="{y}" width="1" height="1" fill="{color}" fill-opacity="{opacity}" />\n'
                else:
                    color = color_info
                    svg_content += f'  <rect x="{x}" y="{y}" width="1" height="1" fill="{color}" />\n'
                
    svg_content += '</svg>'
    return svg_content

def grid_to_png(grid, palette, pixel_size=10):
    height = len(grid)
    width = len(grid[0])
    img = Image.new('RGBA', (width * pixel_size, height * pixel_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for y, row in enumerate(grid):
        for x, char in enumerate(row):
            if char in palette and palette[char] is not None:
                color_info = palette[char]
                color = color_info
                alpha = 255
                
                # Handle hex/tuple logic slightly differently for PIL
                if isinstance(color_info, tuple):
                    hex_color, opacity = color_info
                    color = hex_color
                    alpha = int(opacity * 255)
                
                # Convert hex to RGB
                color = color.lstrip('#')
                rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                
                # Draw rectangle
                draw.rectangle(
                    [x * pixel_size, y * pixel_size, (x + 1) * pixel_size - 1, (y + 1) * pixel_size - 1],
                    fill=rgb + (alpha,)
                )
    return img

def generate_ghost():
    # 24x24 Ghost - Semi-transparent wisp
    grid_str = """
........................
..........####..........
........##WWWW##........
.......#WWWWWWWW#.......
......#WWWWWWWWWW#......
.....#WWWWWWWWWWWW#.....
.....#WWWWWWWWWWWW#.....
....#WWWWWWWWWWWWWW#....
....#WW#WWWWWW#WWWW#....
...#WWW#WWWWWW#WWWWW#...
...#WWW#WWWWWW#WWWWW#...
...#WWWWWWWWWWWWWWWW#...
...#WWWWWWWWWWWWWWWW#...
...#WWWWWWWWWWWWWWWW#...
...#WWWWWWWWWWWWWWWW#...
....#WWWWWWWWWWWWWW#....
....#WWWWWWWWWWWWWW#....
.....#WWWWWWWWWWWW#.....
......#WW##WW##WW#......
.......#W#.#W#.#W#......
........#...#...#.......
........................
........................
........................
"""
    grid = [list(line) for line in grid_str.strip().split('\n')]
    
    palette = {
        '.': None,
        '#': ('#000000', 0.8),  # Slightly transparent outline
        'W': ('#FFFFFF', 0.6),  # Semi-transparent body
        'S': ('#E0E0E0', 0.5)
    }
    
    return grid_to_svg(grid, palette, pixel_size=10)

def generate_logo_1_data():
    # Concept 1: Cute Robot Face (32x32)
    # P: Purple, L: Light Purple, Y: Yellow/Gold eyes, W: White
    
    grid = [['.' for _ in range(32)] for _ in range(32)]
    
    # Face shape (Rounded rect)
    for y in range(4, 28):
        for x in range(4, 28):
            grid[y][x] = 'P' # Face base
            
    # Eyes (Big yellow squares)
    for y in range(10, 16):
        for x in range(8, 13):
            grid[y][x] = 'Y'
        for x in range(19, 24):
            grid[y][x] = 'Y'
            
    # Cheeks (Pink)
    for y in range(18, 21):
        for x in range(6, 9):
            grid[y][x] = 'M'
        for x in range(23, 26):
            grid[y][x] = 'M'
            
    # Mouth (Small smile)
    grid[22][14] = 'W'
    grid[22][15] = 'W'
    grid[22][16] = 'W'
    grid[22][17] = 'W'
    grid[21][13] = 'W'
    grid[21][18] = 'W'
    
    # Antenna
    for y in range(0, 4):
        grid[y][15] = 'G'
        grid[y][16] = 'G'
    
    palette = {
        '.': None,
        'P': '#8B5CF6', # Base Purple
        'M': '#D946EF', # Pink Cheeks
        'Y': '#FFD93D', # Yellow Eyes
        'W': '#FFFFFF', # Mouth
        'G': '#4B5563', # Gray Antenna
    }
    return grid, palette

def generate_logo_2_data():
    # Concept 2: Smiley Magnifier (32x32)
    grid = [['.' for _ in range(32)] for _ in range(32)]
    
    import math
    cx, cy = 14, 14
    radius = 11
    
    # Glass / Face
    for y in range(32):
        for x in range(32):
            d = math.sqrt((x-cx)**2 + (y-cy)**2)
            if d < radius:
                grid[y][x] = 'Y' # Yellow face
            if radius <= d < radius + 2:
                grid[y][x] = 'P' # Purple rim
                
    # Handle
    for i in range(9):
        tx, ty = 22 + i, 22 + i
        if tx < 32 and ty < 32:
            grid[ty][tx] = 'B' # Brown/Dark handle
            if tx+1 < 32: grid[ty][tx+1] = 'B'
            if ty+1 < 32: grid[ty+1][tx] = 'B'
            
    # Face details
    # Eyes
    grid[11][10] = 'K'
    grid[11][18] = 'K'
    grid[12][10] = 'K'
    grid[12][18] = 'K'
    
    # Smile
    for x in range(11, 18):
        grid[18][x] = 'K'
    grid[17][10] = 'K'
    grid[17][18] = 'K'
    
    palette = {
        '.': None,
        'Y': '#FFD93D', # Yellow face
        'P': '#8B5CF6', # Purple rim
        'B': '#4C1D95', # Dark Purple handle
        'K': '#000000', # Black features
    }
    return grid, palette

def generate_logo_3_data():
    # Concept 3: Retro Heart/Shield Check (32x32)
    grid = [['.' for _ in range(32)] for _ in range(32)]
    
    # Draw Heart Base
    heart_str = [
        "..............................",
        "..............................",
        ".....XXXXXX........XXXXXX.....",
        "...XXXXXXXXXX....XXXXXXXXXX...",
        "..XXXXXXXXXXXX..XXXXXXXXXXXX..",
        ".XXXXXXXXXXXXX..XXXXXXXXXXXXX.",
        ".XXXXXXXXXXXXX..XXXXXXXXXXXXX.",
        ".XXXXXXXXXXXXXXXXXXXXXXXXXXXX.",
        ".XXXXXXXXXXXXXXXXXXXXXXXXXXXX.",
        "..XXXXXXXXXXXXXXXXXXXXXXXXXX..",
        "..XXXXXXXXXXXXXXXXXXXXXXXXXX..",
        "...XXXXXXXXXXXXXXXXXXXXXXXX...",
        "....XXXXXXXXXXXXXXXXXXXXXX....",
        ".....XXXXXXXXXXXXXXXXXXXX.....",
        "......XXXXXXXXXXXXXXXXXX......",
        ".......XXXXXXXXXXXXXXXX.......",
        "........XXXXXXXXXXXXXX........",
        ".........XXXXXXXXXXXX.........",
        "..........XXXXXXXXXX..........",
        "...........XXXXXXXX...........",
        "............XXXXXX............",
        ".............XXXX.............",
        "..............XX..............",
        ".............................."
    ]
    
    start_y = 4
    for i, line in enumerate(heart_str):
        for j, char in enumerate(line):
            if char == 'X':
                grid[start_y + i][j+1] = 'P' # Purple base
                
    # Add Check mark in Gold
    # Check shape: V
    check_points = [
        (10, 14), (11, 15), (12, 16), (13, 17), (14, 18), # Down stroke
        (15, 17), (16, 16), (17, 15), (18, 14), (19, 13), (20, 12), (21, 11) # Up stroke
    ]
    # Thicken check
    for cx, cy in check_points:
        for ox in range(3):
            for oy in range(3):
                grid[cy+oy][cx+ox] = 'G'
                
    palette = {
        '.': None,
        'P': '#8B5CF6',
        'G': '#FFD93D'
    }
    return grid, palette

def generate_logo_data():
    # 32x32 Logo - Radar/Target Style
    # Refined Grid with colors
    # P: #8B5CF6 (Purple), L: #D946EF (Magenta/Pink - Highlight), G: #FFD93D (Gold/Yellow)
    
    grid = [['.' for _ in range(32)] for _ in range(32)]
    center_x, center_y = 15.5, 15.5
    import math
    
    for y in range(32):
        for x in range(32):
            dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            # Outer Ring
            if 13.0 <= dist <= 15.0:
                grid[y][x] = 'P'
            # Inner Ring (Scanner line effect)
            if 9.0 <= dist <= 10.0:
                 if (x + y) % 4 == 0: # Dotted effect
                    grid[y][x] = 'L'
            
            # Radar Sweep line (Top Right quadrant)
            angle = math.degrees(math.atan2(center_y - y, x - center_x))
            if 0 <= angle <= 90:
                 if dist < 13.0 and (y - x) % 3 == 0:
                     grid[y][x] = 'S' # Sweep color
                     
    # Center "AI" text
    # A
    for y in range(11, 21):
        grid[y][9] = 'G'
        grid[y][10] = 'G'
        grid[y][13] = 'G'
        grid[y][14] = 'G'
    for x in range(11, 13):
        grid[11][x] = 'G'
        grid[12][x] = 'G'
        grid[15][x] = 'G'
        grid[16][x] = 'G'
        
    # I
    for y in range(11, 21):
        grid[y][18] = 'G'
        grid[y][19] = 'G'              
                     
    palette = {
        '.': None,
        'P': '#8B5CF6', # Violet
        'L': '#C026D3', # Fuchsia
        'S': '#A78BFA', # Lighter Purple
        'G': '#FFD93D', # Gold
        'W': '#FFFFFF'
    }
    return grid, palette

if __name__ == "__main__":
    os.makedirs("public", exist_ok=True)
    
    # Save Ghost as SVG
    with open("public/ghost-pixel.svg", "w") as f:
        f.write(generate_ghost())

    print("Generated semi-transparent ghost icon.")
