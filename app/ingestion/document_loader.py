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


def split_into_chunks(text, max_length=500):
    paragraphs = text.split("\n\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_length:
            chunks.append(para)
        else:
            start = 0
            while start < len(para):
                end = start + max_length
                chunks.append(para[start:end])
                start = end

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
