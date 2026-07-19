import os
import json
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3")

input_folder="data/cleaned_transcripts"
output_folder="data/chunks"

os.makedirs(output_folder,exist_ok=True)

files=os.listdir(input_folder)
for file in files:
    if not file.endswith(".json"):
        continue
    input_path= os.path.join(input_folder,file)
    output_path=os.path.join(output_folder,file)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"{file} already chunked. Skipping...")
        continue

    with open(input_path,"r",encoding="utf-8") as f:
        transcript=json.load(f)
        segments=transcript["segments"]
        for segment in segments:
            text=segment["text"]
            token_count=len(tokenizer.encode(text,add_special_tokens=False))
            segment["token_count"]=token_count
        
        chunks=[]
        current_chunk=[]
        current_tokens=0
        chunk_start=None
        chunk_end=None
        chunk_id=0
        chunk_size=512
        overlap_size=128

        for segment in segments:
            if current_tokens +segment["token_count"]<=chunk_size:
                current_chunk.append(segment)
                current_tokens+= segment["token_count"]
                if chunk_start is None:
                    chunk_start = segment["start"]
                    
                chunk_end= segment["end"]

            else:
                chunk_text=" ".join(segment["text"] for segment in current_chunk)
                chunk={
                    "lecture_name":transcript["lecture_name"],
                    "chunk_id": chunk_id,
                    "start":chunk_start,
                    "end":chunk_end,
                    "text": chunk_text
                }
                chunks.append(chunk)
                chunk_id+=1


                overlap_segments=[]
                overlap_tokens=0
                for s in reversed(current_chunk):
                    overlap_segments.append(s)
                    overlap_tokens+=s["token_count"]

                    if overlap_tokens>=overlap_size:
                        break

                overlap_segments.reverse()
                current_chunk=overlap_segments
                current_tokens=overlap_tokens
                chunk_start=current_chunk[0]["start"]
                chunk_end=current_chunk[-1]["end"]
                
                current_chunk.append(segment)
                current_tokens+=segment["token_count"]
                chunk_end=segment["end"]
                
        
        if current_chunk:
            chunk_text = " ".join(
                s["text"] for s in current_chunk
            )
            chunks.append({
                "lecture_name": transcript["lecture_name"],
                "chunk_id": chunk_id,
                "start": chunk_start,
                "end": chunk_end,
                "text": chunk_text
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=4)

        print(f"{file} -> {len(chunks)} chunks created")