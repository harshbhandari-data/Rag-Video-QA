import os
import json

input_folder="data/transcripts"
output_folder="data/cleaned_taranscripts"

os.makedirs(output_folder,exist_ok=True)

files=[file for file in os.listdir(input_folder) if file.endswith(".json")]

for file in files:
    input_path=os.path.join(input_folder,file)
    output_path=os.path.join(output_folder,file)

    with open(input_path,"r",encoding="utf-8") as f:
        transcript=json.load(f)

    cleaned_segments=[]
    
    for segment in transcript["segments"]:
        text=" ".join(segment["text"].split())

        if not text:
            continue

        cleaned_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": text
        })
    
    transcript["segments"]=cleaned_segments

    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(transcript,f,ensure_ascii=False,indent=4)
    
    print(f"Processed {file}: {len(cleaned_segments)} segments")