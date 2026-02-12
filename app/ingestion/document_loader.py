import os
from pypdf import PdfReader


def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_md(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text.append(content)
    return "\n".join(text)


def split_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def load_documents_from_folder(folder_path: str):
    all_chunks = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):
            text = read_txt(file_path)

        elif filename.endswith(".md"):
            text = read_md(file_path)

        elif filename.endswith(".pdf"):
            text = read_pdf(file_path)

        else:
            continue

        chunks = split_into_chunks(text)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "source": filename
            })

    return all_chunks
