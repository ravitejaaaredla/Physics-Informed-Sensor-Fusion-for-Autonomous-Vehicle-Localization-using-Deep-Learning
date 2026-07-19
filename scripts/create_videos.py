import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

VIS_ROOT = PROJECT_ROOT / "runs" / "full_visualization"
OUTPUT_ROOT = VIS_ROOT / "videos"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

def create_video_from_images(image_dir, output_path, fps=10):
    """Create an MP4 video from a folder of images."""
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        print(f"❌ No PNGs found in {image_dir}")
        return

    # Read the first image to get dimensions
    first_img = cv2.imread(str(image_paths[0]))
    h, w = first_img.shape[:2]

    # Define video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is not None:
            out.write(img)

    out.release()
    print(f"✅ Video saved: {output_path}")

def main():
    print("=" * 80)
    print("CREATING VIDEOS FROM VISUALIZATIONS")
    print("=" * 80)

    # Find all subdirectories in VIS_ROOT (excluding the videos folder)
    for drive_folder in VIS_ROOT.iterdir():
        if not drive_folder.is_dir() or drive_folder.name == "videos":
            continue

        output_path = OUTPUT_ROOT / f"{drive_folder.name}.mp4"
        print(f"\n📂 Processing: {drive_folder.name}")
        create_video_from_images(drive_folder, output_path, fps=10)

    print("\n" + "=" * 80)
    print(f"✅ ALL VIDEOS COMPLETE! Located in: {OUTPUT_ROOT}")
    print("   Open in Finder: open " + str(OUTPUT_ROOT))
    print("=" * 80)

if __name__ == "__main__":
    main()
