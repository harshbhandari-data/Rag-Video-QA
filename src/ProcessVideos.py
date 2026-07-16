import os
import subprocess

video_folder="data/videos"
audio_folder="data/audios"
files=os.listdir(video_folder)
for file in files:
    video_path=os.path.join(video_folder,file)
    audio_name=os.path.splitext(file)[0] + ".mp3"
    audio_path=os.path.join(audio_folder,audio_name)
    if os.path.exists(audio_path):
        print(f"Skipping {file} (already converted)")
        continue
    command=["ffmpeg","-i",video_path,audio_path]
    subprocess.run(command,check=True)

print("Vedio processing complete")