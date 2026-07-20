from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

def load_docx_as_documents(path: str) -> list[Document]:
    docx = DocxDocument(path)
    full_text = "\n\n".join(p.text for p in docx.paragraphs if p.text.strip())
    return [Document(page_content=full_text, metadata={"source": path})]


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)