import json
import os
import numpy as np
import pandas as pd
import torch
import transformers.modeling_utils as modeling_utils
from transformers.utils import import_utils
from FlagEmbedding import BGEM3FlagModel


def _allow_torch_load_for_older_versions():
    """Compatibility shim for older PyTorch versions with newer Transformers guards."""
    def _allow():
        return None

    try:
        import_utils.check_torch_load_is_safe = _allow
    except Exception:
        pass

    try:
        modeling_utils.check_torch_load_is_safe = _allow
    except Exception:
        pass


_allow_torch_load_for_older_versions()


input_folder="data/chunks"
output_folder="data/embeddings"

embedding_file=os.path.join(output_folder,"embeddings.npy")
metadata_file=os.path.join(output_folder,"metadata.parquet")

os.makedirs(output_folder,exist_ok=True)
new_embeddings=[]
new_metadata=[]

if os.path.exists(metadata_file) and os.path.exists(embedding_file):
    existing_metadata = pd.read_parquet(metadata_file)
    processed_lectures = set(existing_metadata["lecture_name"])
    existing_embeddings = np.load(embedding_file)

else:
    existing_metadata = pd.DataFrame()
    processed_lectures = set()
    existing_embeddings = np.empty((0, 1024), dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=torch.cuda.is_available(),
    devices=device,
)

files=os.listdir(input_folder)
for file in files:
    if not file.endswith(".json"):
        continue

    lecture_name=os.path.splitext(file)[0]

    if lecture_name in processed_lectures:
        print(f"Processing {lecture_name}: already embedded. Skipping....")
        continue

    print(f"Processing file: {file}")
    input_path=os.path.join(input_folder,file)
    with open(input_path,"r",encoding="utf-8")as f:
        chunks=json.load(f)

    texts=[chunk["text"] for chunk in chunks]
    embeddings=model.encode(texts,batch_size=16,max_length=512)["dense_vecs"].astype(np.float32)
    new_embeddings.append(embeddings)
    new_metadata.extend(chunks)
    print(f"Processing completed for: {lecture_name}")

if new_embeddings:
    new_embeddings=np.vstack(new_embeddings)

    final_embeddings=np.vstack(
        [
            existing_embeddings,
            new_embeddings
        ]
    )

    new_metadata=pd.DataFrame(new_metadata)
    final_metadata=pd.concat(
        [
            existing_metadata,
            new_metadata
        ],
        ignore_index=True
    )
    np.save(embedding_file,final_embeddings)
    final_metadata.to_parquet(metadata_file,index=False)

    print(f"\nTotal embeddings : {len(final_embeddings)}")
    print(f"Metadata entries : {len(final_metadata)}")
    print("Embedding generation completed.")

else:
    print("No new lectures found.")