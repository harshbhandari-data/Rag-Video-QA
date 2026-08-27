import streamlit as st
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

    if hasattr(import_utils, "check_torch_load_is_safe"):
        import_utils.check_torch_load_is_safe = allow

    if hasattr(modeling_utils, "check_torch_load_is_safe"):
        modeling_utils.check_torch_load_is_safe = allow


def format_time(seconds):

    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


_allow_torch_load_for_older_versions()


@st.cache_resource
def load_resources():

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    index_file = os.path.join(
        BASE_DIR,
        "data",
        "vector_store",
        "faiss.index"
    )

    metadata_file = os.path.join(
        BASE_DIR,
        "data",
        "embeddings",
        "metadata.parquet"
    )

    index = faiss.read_index(index_file)

    metadata = pd.read_parquet(metadata_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_bf16=torch.cuda.is_available(),
        devices=device
    )

    return index, metadata, model


index, metadata, model = load_resources()


lecture_count = metadata["lecture_name"].nunique()


st.title("🎓 Lecture RAG Assistant")

st.subheader("📚 Course: 100 Days of Machine Learning By campusX")

st.write(
    "Ask questions based on lectures from the 100 Days of Machine Learning course."
)

st.write(
    f"🎥 Currently, this RAG system can answer questions from **{lecture_count} lectures**."
)


st.divider()


query = st.text_input(
    "Enter your question"
)


if st.button("Ask"):

    if query:

        with st.spinner("🔍 Searching through lectures..."):

            query_embedding = model.encode(
                [query],
                batch_size=1,
                max_length=512
            )["dense_vecs"].astype(np.float32)

            faiss.normalize_L2(query_embedding)

            k = 15

            scores, indices = index.search(
                query_embedding,
                k
            )

            contexts = []

            for index_id in indices[0]:

                chunk = metadata.iloc[index_id]

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


        prompt = f"""
Answer the user's question using only the provided context.

Give a clear and detailed explanation using the important
points and examples from the context.

If the context does not contain the answer, say that you
could not find the answer instead of guessing.

Context:
{context}

Question:
{query}

Answer:
"""


        with st.spinner("🤖 Generating answer..."):

            response = ollama.chat(
                model="qwen3.5:9b",
                think=False,
                options={
                    "temperature": 0,
                    "num_predict": 1024
                },
                messages=[
                    {
                        "role": "user",
                        "content": prompt + "\nBe detailed but avoid repetition."
                    }
                ]
            )


        answer = response["message"]["content"]


        st.divider()

        st.subheader("🤖 Answer")

        st.write(answer)


        st.divider()

        st.subheader("📖 Sources")

        for chunk in top_contexts:

            start_time = format_time(chunk["start"])
            end_time = format_time(chunk["end"])

            st.write(
                f"**🎥 Lecture:** {chunk['lecture_name']}"
            )

            st.write(
                f"**🕒 Timestamp:** {start_time} - {end_time}"
            )

            st.divider()


    else:

        st.warning("Please enter a question.")