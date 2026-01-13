"""
Use this script to resample videos to a uniform frame rate and create required synchronization ('sync') files.

This script:
1. Reads video files from different camera subdirectories
2. Resamples all videos to a specified target frame rate
3. Saves the resampled videos
4. Generates synchronization files for the resampled videos

Use of resampleAndSyncFiles.py:
    python resampleAndSyncFiles.py path_to_video_folder target_frame_rate num_landmarks

path_to_video_folder: string, the path to your project main video folder, which
    contains separate subfolders for each camera. These subfolders contain the
    video files.
target_frame_rate: int or float, the target frame rate to resample all videos to
num_landmarks: number of landmarks you plan to label/track

Resampled videos will be written to a `resampled` directory in the video folder parent directory.
Sync files will be written to a `sync` directory in the video folder parent directory.
"""

import imageio
import numpy as np
import scipy.io as sio
import os
import sys
from tqdm import tqdm

_VALID_EXT = ["mp4", "avi"]

def get_vid_paths(dir_):
    """Get all valid video file paths from a directory"""
    vids = os.listdir(dir_)
    vids = [vd for vd in vids if vd.split(".")[-1] in _VALID_EXT]
    vids = [os.path.join(dir_, vd) for vd in vids]
    return vids

def resample_video(input_path, output_path, source_fps, target_fps):
    """
    Resample a video to a new frame rate
    
    Args:
        input_path: path to input video
        output_path: path to save resampled video
        source_fps: original frame rate
        target_fps: target frame rate
    """
    print(f"Resampling {os.path.basename(input_path)} from {source_fps:.2f} fps to {target_fps:.2f} fps...")
    
    # Read input video
    reader = imageio.get_reader(os.path.abspath(input_path))
    source_fps_actual = reader.get_meta_data()['fps']
    total_frames = reader.count_frames()
    
    # Calculate frame indices to extract (nearest neighbor resampling)
    num_output_frames = int(np.ceil(total_frames * target_fps / source_fps_actual))
    frame_indices = np.linspace(0, total_frames - 1, num_output_frames).astype(int)
    
    # Write resampled video
    writer = imageio.get_writer(output_path, fps=target_fps)
    
    for idx in tqdm(frame_indices, desc="Processing frames"):
        frame = reader.get_data(idx)
        writer.append_data(frame)
    
    writer.close()
    reader.close()
    
    print(f"Resampled video saved to {output_path}")
    return num_output_frames

if __name__ == "__main__":
    vidpath = sys.argv[1]
    target_fps = float(sys.argv[2])
    num_landmarks = int(sys.argv[3])
    
    # Setup output paths
    parent_dir = os.path.dirname(vidpath.rstrip(os.sep))
    resampled_outpath = os.path.join(parent_dir, "resampled", os.path.basename(vidpath))
    sync_outpath = os.path.join(parent_dir, "sync", os.path.basename(vidpath))
    
    if not os.path.exists(resampled_outpath):
        os.makedirs(resampled_outpath)
        print(f"Created resampled directory: {resampled_outpath}")
    
    if not os.path.exists(sync_outpath):
        os.makedirs(sync_outpath)
        print(f"Created sync directory: {sync_outpath}")
    
    print(f"Reading videos from {vidpath}...")
    print(f"Target frame rate: {target_fps} fps\n")
    
    # Find camera directories
    dirs = os.listdir(vidpath)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(vidpath, d))]
    dirs = [d for d in dirs if d not in ['data', 'Camera0']]
    print(f"Found the following cameras: {dirs}\n")
    
    dirs = [os.path.join(vidpath, d) for d in dirs]
    
    # Step 1: Resample all videos
    print("=" * 60)
    print("STEP 1: Resampling videos")
    print("=" * 60 + "\n")
    
    camnames = []
    resampled_framecount = []
    
    for d in dirs:
        vids = get_vid_paths(d)
        if len(vids) == 0:
            print("Traversing video subdirectory")
            d = os.path.join(d, os.listdir(d)[0])
            vids = get_vid_paths(d)
        
        cname = os.path.basename(d.rstrip(os.sep))
        camnames.append(cname)
        cam_resampled_dir = os.path.join(resampled_outpath, cname)
        
        if not os.path.exists(cam_resampled_dir):
            os.makedirs(cam_resampled_dir)
        
        total_resampled_frames = 0
        
        for i, vid_path in enumerate(vids):
            output_vid_path = os.path.join(cam_resampled_dir, os.path.basename(vid_path))
            
            # Get original fps
            reader = imageio.get_reader(os.path.abspath(vid_path))
            source_fps = reader.get_meta_data()['fps']
            reader.close()
            
            # Resample video
            num_frames = resample_video(vid_path, output_vid_path, source_fps, target_fps)
            total_resampled_frames += num_frames
        
        resampled_framecount.append(total_resampled_frames)
        print(f"Total resampled frames for {cname}: {total_resampled_frames}\n")
    
    # Check that all videos have same length after resampling
    if np.sum(resampled_framecount) // len(resampled_framecount) != resampled_framecount[0]:
        raise Exception("Your resampled videos are not the same length. Check your source videos.")
    
    # Step 2: Generate sync files
    print("=" * 60)
    print("STEP 2: Generating sync files")
    print("=" * 60 + "\n")
    
    fp = 1000.0 / target_fps  # frame period in ms
    
    data_frame = np.arange(resampled_framecount[0]).astype("float64")
    data_sampleID = data_frame * fp + 1
    data_2d = np.zeros((resampled_framecount[0], 2 * num_landmarks))
    data_3d = np.zeros((resampled_framecount[0], 3 * num_landmarks))
    
    checkf = os.listdir(sync_outpath)
    for cname in camnames:
        fname = cname + "_sync.mat"
        outfile = os.path.join(sync_outpath, fname)
        
        if fname in checkf:
            ans = ""
            while ans != "y" and ans != "n":
                print(f"{fname} already exists. Overwrite (y/n)?")
                ans = input().lower()
            
            if ans == "n":
                print("Ok, skipping.")
                continue
        
        print(f"Writing {outfile}")
        sio.savemat(
            outfile,
            {
                "data_frame": data_frame[:, np.newaxis],
                "data_sampleID": data_sampleID[:, np.newaxis],
                "data_2d": data_2d,
                "data_3d": data_3d,
            },
        )
    
    print("\nAll tasks completed!")
    print(f"Resampled videos saved to: {resampled_outpath}")
    print(f"Sync files saved to: {sync_outpath}")
