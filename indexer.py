import os
import logging
import sys
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
load_dotenv()

Settings.llm = Gemini(model_name="models/gemini-1.5-flash-latest", api_key=os.getenv("GOOGLE_API_KEY"))
Settings.embed_model = GeminiEmbedding(model_name="models/text-embedding-004", api_key=os.getenv("GOOGLE_API_KEY"))

PERSIST_DIR = "./storage"

def create_index():
    print("Indekslash jarayoni boshlandi...")
    documents = SimpleDirectoryReader("./data").load_data()
    print(f"{len(documents)} ta hujjat yuklandi.")

    parser = MarkdownNodeParser(include_metadata=True)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Hújjetler {len(nodes)} dana Markdown tiykarındaǵı bólekke ajıratıldı.")

    index = VectorStoreIndex(nodes)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Indeks muvaffaqiyatli yaratildi va '{PERSIST_DIR}' papkasida saqlandi.")

if __name__ == "__main__":
    create_index()