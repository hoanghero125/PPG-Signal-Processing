# dataset/data_loader/CustomLoader.py

import os
import numpy as np
import pandas as pd
import cv2
from dataset.data_loader.BaseLoader import BaseLoader
from scipy import interpolate
import glob

class CustomLoader(BaseLoader):
    """Custom loader for your baseline dataset with PPG CSV files"""
    
    def __init__(self, name, data_path, config_data, device=None):
        """Initialize the data loader"""
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data directories under the path"""
        data_dirs = glob.glob(data_path + os.sep + "*")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        
        # Filter to only include directories
        data_dirs = [d for d in data_dirs if os.path.isdir(d)]
        
        dirs = [{"index": os.path.basename(data_dir), "path": data_dir} for data_dir in data_dirs]
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values."""
        if begin == 0 and end == 1:
            return data_dirs

        file_num = len(data_dirs)
        choose_range = range(int(begin * file_num), int(end * file_num))
        data_dirs_new = []

        for i in choose_range:
            data_dirs_new.append(data_dirs[i])

        return data_dirs_new
        
    @staticmethod
    def read_video(video_file):
        """Read video file and return frames as numpy array"""
        frames = []
        cap = cv2.VideoCapture(video_file)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_file}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames read from: {video_file}")
        
        # Convert to numpy array: [num_frames, height, width, channels]
        frames = np.array(frames)
        return frames
    
    @staticmethod
    def read_wave(bvp_file):
        """Read PPG signal from CSV file"""
        # Read the CSV
        df = pd.read_csv(bvp_file)
        
        # Use green channel (best for PPG)
        if 'green' not in df.columns:
            raise ValueError(f"'green' column not found in {bvp_file}")
        
        # Extract PPG signal
        bvp_signal = df['green'].values.astype(np.float32)
        
        # Handle timestamps
        if 'timestamp' in df.columns:
            timestamps = pd.to_datetime(df['timestamp'])
            # Convert to seconds from start
            time_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
        else:
            # Assume 25Hz if no timestamps
            time_seconds = np.arange(len(bvp_signal)) / 25.0
        
        # Calculate sampling frequency
        if len(time_seconds) > 1:
            fs = len(bvp_signal) / time_seconds[-1]
        else:
            fs = 25.0  # Default to 25Hz
        
        # Return in the format expected by BaseLoader
        return bvp_signal
    
    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Preprocess a single subject (called by parent class in multiprocessing)"""
        saved_filename = data_dirs[i]['index']
        subject_path = data_dirs[i]['path']
        
        try:
            # Find video file
            video_files = [
                f for f in os.listdir(subject_path) 
                if f.endswith(('.mp4', '.avi', '.mov'))
            ]
            
            if len(video_files) == 0:
                print(f"   No video in {saved_filename}")
                return
            
            video_file = os.path.join(subject_path, video_files[0])
            
            # Find PPG file
            ppg_file = os.path.join(subject_path, 'ppg.csv')
            if not os.path.exists(ppg_file):
                print(f"   No ppg.csv in {saved_filename}")
                return
            
            # Read video
            frames = self.read_video(video_file)
            
            # Read PPG (using parent's method if needed)
            if config_preprocess.USE_PSUEDO_PPG_LABEL:
                bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
            else:
                bvps = self.read_wave(ppg_file)
            
            # Resample PPG to match video frame count
            bvps = self.resample_ppg(bvps, len(frames))
            
            # Preprocess using parent class method
            frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
            
            # Save using parent class method
            input_name_list, label_name_list = self.save_multi_process(
                frames_clips, bvps_clips, saved_filename
            )
            
            file_list_dict[i] = input_name_list
            
        except Exception as e:
            print(f"   Error processing {saved_filename}: {e}")
            file_list_dict[i] = []
    
    @staticmethod
    def resample_ppg(ppg_signal, target_length):
        """Resample PPG signal to match video frame count"""
        original_length = len(ppg_signal)
        
        if original_length == target_length:
            return ppg_signal
        
        # Create interpolation function
        x_old = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, target_length)
        
        interp_func = interpolate.interp1d(x_old, ppg_signal, kind='linear')
        ppg_resampled = interp_func(x_new)
        
        return ppg_resampled