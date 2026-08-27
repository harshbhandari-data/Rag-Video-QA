import pandas as pd
import numpy as np
import os 
import faiss
import torch
import transformers.modeling_utils as modeling_utils
from transformers.utils import import_utils
from FlagEmbedding import BGEM3FlagModel
import ollama

def _allow_torch_load_for_older_versions():
    def allow():
        return None
    if hasattr(import_utils,"check_torch_load_is_safe"):
        import_utils.check_torch_load_is_safe=allow

    if hasattr(modeling_utils,"check_torch_load_is_safe"):
        modeling_utils.check_torch_load_is_safe=allow

def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"

_allow_torch_load_for_older_versions()

index_file=os.path.join("data","vector_store","faiss.index")
metadata_file=os.path.join("data","embeddings","metadata.parquet")

index=faiss.read_index(index_file)
print(f"Faiss vectors : {index.ntotal}")

metadata=pd.read_parquet(metadata_file)
print(f"Metadata rows : {len(metadata)}")

device="cuda"if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")

model=BGEM3FlagModel(
    "BAAI/bge-m3",
    use_bf16=torch.cuda.is_available(),
    devices=device,
    )

query=input("\n Enter your question: ")

query_embedding= model.encode(
    [query],
    batch_size=1,
    max_length=512
)["dense_vecs"].astype(np.float32)

faiss.normalize_L2(query_embedding)

k=15

scores,indices=index.search(query_embedding,k)


contexts=[]

for index_id in indices[0]:
    chunk=metadata.iloc[index_id]

    contexts.append({
        "text": chunk["text"],
        "lecture_name": chunk["lecture_name"],
        "start": chunk["start"],
        "end": chunk["end"]
    })

top_contexts = contexts[:8]

context = "\n\n".join(
    chunk["text"][:1600]
    for chunk in top_contexts
)

prompt= f"""
Answer the user's question using only the provided context. Give a clear, detailed explanation with the important points, examples, and steps when they are present in the context. If the context does not contain the answer, say that you could not find it instead of guessing.

Context:
{context}

Question:
{query}

Answer:
"""

response=ollama.chat(
    model="qwen3.5:9b",
    think=False,
    options={
        "temperature": 0,
        "num_predict": 1024,
    },
    messages=[
        {
            "role": "user",
            "content": prompt + "\nBe detailed but avoid repetition."
        }
    ]
)
print("\n========== Answer ==========")

print(response["message"]["content"])

print("\n========== Sources ==========")

for chunk in top_contexts:
    start_time = format_time(chunk["start"])
    end_time = format_time(chunk["end"])

    print(f"\nLecture: {chunk['lecture_name']}")
    print(f"Timestamp: {start_time} - {end_time}")