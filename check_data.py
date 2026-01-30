# check_data.py
from pathlib import Path
import pandas as pd
import cv2

def verify_raw_data(raw_data_path="raw_data"):
    """Verify your raw data is ready"""
    raw_data = Path(raw_data_path)
    
    print("=" * 60)
    print("VERIFYING RAW DATA")
    print("=" * 60)
    
    subjects = [d for d in raw_data.iterdir() if d.is_dir()]
    print(f"\n✅ Found {len(subjects)} subjects\n")
    
    all_good = True
    
    for subject in subjects:
        print(f"📁 {subject.name}")
        
        # Check video
        videos = list(subject.glob("*.mp4")) + list(subject.glob("*.avi"))
        if not videos:
            print(f"   ❌ No video file found")
            all_good = False
            continue
        
        video = videos[0]
        print(f"   📹 Video: {video.name}")
        
        # Test video
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print(f"   ❌ Cannot open video")
            all_good = False
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"      {width}x{height}, {fps}fps, {frames} frames ({frames/fps:.1f}s)")
            cap.release()
        
        # Check PPG
        ppg_file = subject / "ppg.csv"
        if not ppg_file.exists():
            print(f"   ❌ No ppg.csv found")
            all_good = False
            continue
        
        print(f"   📊 PPG: ppg.csv")
        
        try:
            df = pd.read_csv(ppg_file)
            if 'green' not in df.columns or 'timestamp' not in df.columns:
                print(f"   ❌ Missing required columns (green, timestamp)")
                all_good = False
            else:
                print(f"      {len(df)} samples (~{len(df)/25:.1f}s at 25Hz)")
        except Exception as e:
            print(f"   ❌ Error reading ppg.csv: {e}")
            all_good = False
        
        print()
    
    if all_good:
        print("🎉 All data verified! Ready to proceed.")
    else:
        print("⚠️  Please fix issues above before proceeding.")
    
    return all_good

# Run verification
if __name__ == "__main__":
    verify_raw_data("raw_data")