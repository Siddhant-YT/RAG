"""
app/tab_multimodal.py
----------------------
Tab 5: Multimodal RAG
Demonstrates RAG over documents that contain images.
The user can upload an image; Gemini Vision generates a text caption.
That caption is stored in FAISS alongside text chunks.
When retrieved, the image is sent directly to Gemini for visual QA.
"""

import streamlit as st
import tempfile
import os

from core.models import load_gemini_llm, load_embedding_model, load_groq_llm
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage


def describe_image_with_gemini(image_path: str, gemini_llm) -> str:
    """
    Use Gemini Vision to generate a detailed text description of an image.
    This description is then embedded in FAISS, making image content text-searchable.
    """
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()

        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

        from PIL import Image
        image = Image.open(image_path)

        response = model.generate_content([
            "Describe all visual elements, text, charts, data, diagrams, and any text visible in this image "
            "in thorough detail. This description will be used in a knowledge base for retrieval.",
            image
        ])
        return response.text
    except Exception as e:
        return f"[Caption generation error: {e}]"


def answer_image_question(image_path: str, question: str) -> str:
    """
    Directly answer a question about an image using Gemini Vision.
    Used when an image document is retrieved in response to a query.
    """
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        load_dotenv()

        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

        from PIL import Image
        image = Image.open(image_path)

        response = model.generate_content([question, image])
        return response.text
    except Exception as e:
        return f"[Image QA error: {e}]"


def render():
    """Render the Multimodal RAG tab."""
    st.header("Multimodal RAG")
    st.write(
        "Standard RAG handles text only. Multimodal RAG extends this to images. "
        "Upload an image and a text document. The system captions the image using Gemini Vision, "
        "stores the caption in the same vector store as text chunks, and handles queries "
        "that require understanding both text and image content."
    )

    st.info(
        "Requires GOOGLE_API_KEY in your .env file. "
        "Get a free key at https://aistudio.google.com/app/apikey"
    )

    embedding_model = load_embedding_model()
    groq_llm = load_groq_llm()

    st.divider()

    # ---- Step 1: Upload Image ----
    st.subheader("Step 1: Upload an Image")
    st.write(
        "Upload any image (diagram, chart, screenshot, photo). "
        "Gemini Vision will describe it so it can be searched via text queries."
    )

    uploaded_image = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        key="mm_image",
    )

    image_caption = ""
    image_path_tmp = None

    if uploaded_image:
        # Display the image
        st.image(uploaded_image, caption=uploaded_image.name, use_column_width=True)

        # Save to temp file for Gemini processing
        suffix = os.path.splitext(uploaded_image.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_image.read())
            image_path_tmp = tmp.name

        if st.button("Generate Image Caption with Gemini Vision"):
            with st.spinner("Gemini is analyzing the image..."):
                image_caption = describe_image_with_gemini(image_path_tmp, None)
                st.session_state["mm_image_caption"] = image_caption
                st.session_state["mm_image_path"] = image_path_tmp

        if st.session_state.get("mm_image_caption"):
            st.subheader("Generated Caption")
            st.success(st.session_state["mm_image_caption"])

    st.divider()

    # ---- Step 2: Add text context ----
    st.subheader("Step 2: Add Text Context (Optional)")
    st.write("Add any text that provides background for the image.")

    text_context = st.text_area(
        "Background text",
        placeholder="e.g. This is a system architecture diagram showing the RAG pipeline components...",
        height=100,
    )

    st.divider()

    # ---- Step 3: Build multimodal knowledge base ----
    st.subheader("Step 3: Build Multimodal Knowledge Base")

    if st.button("Build Multimodal Knowledge Base", type="primary"):
        all_docs = []

        # Add text context if provided
        if text_context.strip():
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
            text_doc = Document(
                page_content=text_context.strip(),
                metadata={"source": "text_context", "content_type": "text"}
            )
            text_chunks = splitter.split_documents([text_doc])
            all_docs.extend(text_chunks)

        # Add image caption as a document
        caption = st.session_state.get("mm_image_caption", "")
        if caption:
            img_doc = Document(
                page_content=caption,
                metadata={
                    "source": uploaded_image.name if uploaded_image else "image",
                    "content_type": "image",
                    "image_path": st.session_state.get("mm_image_path", ""),
                }
            )
            all_docs.append(img_doc)

        if not all_docs:
            st.warning("Please upload an image or add text context first.")
        else:
            with st.spinner("Building multimodal knowledge base..."):
                mm_vs = FAISS.from_documents(all_docs, embedding_model)
                st.session_state["mm_vectorstore"] = mm_vs
                st.session_state["mm_docs"] = all_docs
            st.success(f"Multimodal knowledge base built with {len(all_docs)} document(s).")

    st.divider()

    # ---- Step 4: Query ----
    st.subheader("Step 4: Ask a Question")
    st.write(
        "Ask a question about the content. If the image is the best match, "
        "Gemini Vision will directly answer from the image."
    )

    if not st.session_state.get("mm_vectorstore"):
        st.info("Build the multimodal knowledge base above first.")
        return

    mm_vs = st.session_state["mm_vectorstore"]

    query = st.text_input(
        "Query",
        placeholder="e.g. What does the diagram show?",
        key="mm_query",
    )

    if st.button("Search and Answer", disabled=not query.strip()):
        with st.spinner("Retrieving and generating..."):
            retrieved = mm_vs.similarity_search(query.strip(), k=3)

        st.subheader("Retrieved Documents")

        text_contexts = []
        image_answers = []

        for doc in retrieved:
            content_type = doc.metadata.get("content_type", "text")
            source = doc.metadata.get("source", "unknown")

            if content_type == "image":
                img_path = doc.metadata.get("image_path", "")
                st.write(f"Image document matched: **{source}**")

                if img_path and os.path.exists(img_path):
                    # Ask Gemini Vision to answer directly from the image
                    img_answer = answer_image_question(img_path, query.strip())
                    image_answers.append(img_answer)
                    with st.expander("Gemini Vision answer from image"):
                        st.write(img_answer)
                else:
                    # Fall back to caption text
                    text_contexts.append(doc.page_content)
            else:
                text_contexts.append(doc.page_content)
                with st.expander(f"Text chunk from: {source}"):
                    st.write(doc.page_content)

        # Build combined context and generate final answer
        full_context = "\n\n".join(text_contexts)
        if image_answers:
            full_context += "\n\nVisual content analysis:\n" + "\n".join(image_answers)

        if full_context.strip():
            messages = [
                SystemMessage(content=(
                    "You are a knowledgeable assistant. The context may include text and image analysis. "
                    "Answer based on all provided context."
                )),
                HumanMessage(content=f"Context:\n{full_context}\n\nQuestion: {query.strip()}")
            ]
            with st.spinner("Generating final answer..."):
                answer = groq_llm.invoke(messages).content

            st.divider()
            st.subheader("Final Answer")
            st.write(answer)
        else:
            st.warning("No context retrieved.")
