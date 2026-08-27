import pandas as pd
import numpy as np
import os
import faiss
import torch
import transformers.modeling_utils as modeling_utils
from transformers.utils import import_utils
from FlagEmbedding import BGEM3FlagModel


def _allow_torch_load_for_older_versions():
    def _allow():
        return None

    if hasattr(import_utils, "check_torch_load_is_safe"):
        import_utils.check_torch_load_is_safe = _allow

    if hasattr(modeling_utils, "check_torch_load_is_safe"):
        modeling_utils.check_torch_load_is_safe = _allow


_allow_torch_load_for_older_versions()


index_file=os.path.join("data","vector_store","faiss.index")
metadata_file= os.path.join("data","embeddings","metadata.parquet")

index=faiss.read_index(index_file)
print(f"Faiss vectors: {index.ntotal}")

metadata=pd.read_parquet(metadata_file)
print(f"metadata rows: {len(metadata)}")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model=BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=torch.cuda.is_available(),
    devices=device,
)
 
query= input("\n Enter your question : ")

query_embedding=model.encode(
    [query],
    batch_size=1,
    max_length=512
)["dense_vecs"].astype(np.float32)

faiss.normalize_L2(query_embedding)

k=5    # top 5 matching chunks we are taking
score,indices=index.search(query_embedding,k)


print("\n=========== Retrived Chunks ===========\n")

for rank,(score,index_id) in enumerate(
    zip(score[0],indices[0]),
    start=1
):
    chunk=metadata.iloc[index_id]

    print(f"Result {rank}")
    print(f"Score     : {score:.4f}")
    print(f"Lecture   : {chunk['lecture_name']}")
    print(f"Chunk ID     : {chunk['chunk_id']}")
    print(f"Start       : {chunk['start']}")
    print(f"End         : {chunk['end']}")
    print(f"Text        : {chunk['text']}")
    print("-" * 80)