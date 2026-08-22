import os
import numpy as np
import faiss

embedding_file="data/embeddings/embeddings.npy"
output_folder="data/vector_store"
index_file=os.path.join(output_folder,"faiss.index")

os.makedirs(output_folder,exist_ok=True)

embeddings= np.load(embedding_file).astype(np.float32)

print(f"Total embeddings in file: {len(embeddings)}")

if(os.path.exists(index_file)):
    index=faiss.read_index(index_file)
    print(f"Existing FAISS vectors : {index.ntotal}")

else:
    dimensions=embeddings.shape[1]
    index=faiss.IndexFlatIP(dimensions)
    print("No existing FAISS index found. Creating a new one...")

processed_count=index.ntotal
total_count=len(embeddings)

if processed_count>total_count:
    raise ValueError(
        "FAISS index contains more vector than embeddings ...."
    )
if processed_count==total_count:
    print("FAISS index is up to date ...")

else:
    new_embeddings=embeddings[processed_count:]
    print(f"New embeddings to index: {len(new_embeddings)}")


    faiss.normalize_L2(new_embeddings)

    index.add(new_embeddings)

    faiss.write_index(index,index_file)
    print(f"FAISS index updated. ")
    print(f"Total vectors in FAISS : {index.ntotal}")
