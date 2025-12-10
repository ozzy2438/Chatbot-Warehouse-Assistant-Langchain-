#!/usr/bin/env python3
"""
Create GIF from linkedin_blueprint_v2.html animation
"""
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
import io

def create_gif():
    # Setup Chrome
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')

    print("Starting Chrome...")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
    )

    # Get absolute path to HTML file
    html_path = os.path.abspath('linkedin_blueprint_v2.html')
    file_url = f'file://{html_path}'

    print(f"Loading: {file_url}")
    driver.get(file_url)

    # Wait for page to load
    time.sleep(2)

    # Capture frames
    frames = []
    total_duration = 12  # seconds (terminal animation is ~11 seconds)
    fps = 4  # frames per second
    total_frames = total_duration * fps

    print(f"Capturing {total_frames} frames...")

    for i in range(total_frames):
        # Take screenshot
        screenshot = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(screenshot))

        # Resize for LinkedIn (recommended: 1200x628 or similar)
        img = img.resize((1200, 675), Image.Resampling.LANCZOS)

        frames.append(img)

        if (i + 1) % 10 == 0:
            print(f"  Frame {i + 1}/{total_frames}")

        time.sleep(1 / fps)

    driver.quit()

    # Create GIF
    output_path = 'linkedin_etl_pipeline.gif'
    print(f"\nCreating GIF: {output_path}")

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),  # milliseconds per frame
        loop=0,  # infinite loop
        optimize=True
    )

    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nDone! GIF created: {output_path}")
    print(f"Size: {size_mb:.2f} MB")

    # LinkedIn limit is 5MB for images
    if size_mb > 5:
        print("\nNote: File is larger than 5MB. Creating optimized version...")
        create_optimized_gif(frames, fps)

def create_optimized_gif(frames, fps):
    """Create smaller GIF by reducing colors and frames"""
    # Take every other frame
    reduced_frames = frames[::2]

    # Convert to palette mode for smaller size
    palette_frames = []
    for img in reduced_frames:
        # Convert to P mode with 128 colors
        img_p = img.convert('P', palette=Image.ADAPTIVE, colors=128)
        palette_frames.append(img_p)

    output_path = 'linkedin_etl_pipeline_optimized.gif'
    palette_frames[0].save(
        output_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=int(2000 / fps),  # slower since we have fewer frames
        loop=0,
        optimize=True
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Optimized GIF: {output_path} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    create_gif()
