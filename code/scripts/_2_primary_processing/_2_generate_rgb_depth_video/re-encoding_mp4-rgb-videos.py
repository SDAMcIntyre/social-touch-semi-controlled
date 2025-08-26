import os
import subprocess
from pathlib import Path

# --- CONFIGURATION ---

# 1. Set the root folder to search for videos.
#    Use a raw string (r"...") on Windows to handle backslashes correctly.
#    Example Windows: r"C:\Users\YourUser\Videos"
#    Example macOS/Linux: "/home/youruser/videos"
ROOT_FOLDER = Path(r"F:\\OneDrive - Linköpings universitet\\_Teams\\Social touch Kinect MNG\\data\\semi-controlled\\1_primary\\kinect")

# 2. (Optional) If ffmpeg is not in your system's PATH, specify the full path.
#    Example Windows: FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_PATH = "ffmpeg"  # This assumes ffmpeg is in the system PATH

# 3. Define the video file extensions to look for.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}

# --- MAIN SCRIPT ---

def re_encode_videos(root_dir: Path):
    """
    Recursively finds videos, renames the original, and re-encodes it
    to the original filename.
    """
    if not root_dir.is_dir():
        print(f"❌ Error: Folder not found at '{root_dir}'")
        return

    print(f"🚀 Starting video processing in '{root_dir}'...")

    # Use os.walk to recursively go through all directories
    for current_folder, _, files in os.walk(root_dir):
        for filename in files:
            original_path = Path(current_folder) / filename

            # --- Check if the file should be processed ---
            if original_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue  # Not a video file

            if "_no-re-encoding" in original_path.stem:
                continue  # This is a renamed original, skip it

            print("-" * 60)
            print(f"🔍 Found video: {original_path}")

            renamed_path = original_path.with_stem(original_path.stem + "_no-re-encoding")
            
            if renamed_path.exists():
                print(f"⚠️  Skipping: Renamed original '{renamed_path.name}' already exists.")
                continue

            try:
                # 1. Rename the original file
                original_path.rename(renamed_path)
                print(f"   -> Renamed original to '{renamed_path.name}'")

                # 2. Build and run the FFmpeg command
                command = [
                    FFMPEG_PATH,
                    "-i", str(renamed_path),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "22",
                    "-g", "1",
                    str(original_path)  # Output is the original filename
                ]

                print("   -> Running FFmpeg re-encoding...")
                subprocess.run(
                    command,
                    check=True,       # Raises an error if FFmpeg fails
                    capture_output=True, # Hides FFmpeg's console output
                    text=True
                )
                print(f"✅ Successfully re-encoded to '{original_path.name}'")

            except subprocess.CalledProcessError as e:
                print(f"❌ FFmpeg Error for '{original_path.name}':")
                print(f"   {e.stderr.strip()}")
                print("   -> Restoring original filename...")
                renamed_path.rename(original_path) # Rename back on failure

            except Exception as e:
                print(f"❌ An unexpected error occurred with '{original_path.name}': {e}")
                # Attempt to restore the filename if possible
                if renamed_path.exists() and not original_path.exists():
                    renamed_path.rename(original_path)

    print("\n" + "=" * 60)
    print("🎉 Processing complete.")

if __name__ == "__main__":
    re_encode_videos(ROOT_FOLDER)