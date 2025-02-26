from docling.document_converter import DocumentConverter, PdfFormatOption
from pathlib import Path
from typing import Iterable
from dataclasses import dataclass
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.chunking import HybridChunker
from docling.datamodel.document import ConversionResult, ConversionStatus
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobLeaseClient
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
import uuid
import os
import dotenv
from ..shared.storage import Container

dotenv.load_dotenv()

IMAGE_RESOLUTION_SCALE = 2.0
OUTPUT_DIR = Path("output")
DOCUMENT_CONTAINER = "documents"
PROCESSED_DOCUMENT_CONTAINER = 'processed-documents'
MAX_TOKENS = 64

upload_results = os.getenv("UPLOAD_RESULTS", "false").lower() == "true"
storage_url = os.getenv("STORAGE_ACCOUNT_URL")
chunking_enabled = os.getenv("CHUNKING_ENABLED", "false").lower() == "true"

embeddings_model = AzureOpenAIEmbeddings(    
    azure_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    model= os.getenv("AZURE_OPENAI_EMBEDDING_MODEL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

@dataclass
class Document:
    id: str
    page_content: str
    metadata: dict

def main():
    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.CPU)

    options = PdfPipelineOptions()
    options.images_scale = IMAGE_RESOLUTION_SCALE
    options.generate_picture_images = True
    options.generate_page_images = True
    options.accelerator_options = accelerator_options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF:PdfFormatOption(pipeline_options=options),
        },
    )

    search_index = create_search_index()

    dir = Path(OUTPUT_DIR)
    dir.mkdir(parents=True, exist_ok=True)

    container = Container(storage_url, DefaultAzureCredential(), DOCUMENT_CONTAINER)
    container.create_container()
    files = container.get_files()

    for file in files:
        file_output_path = Path(OUTPUT_DIR, file.name)

        if file.is_locked():
            continue

        file.lease()
        file.download(file_output_path)

        conversion_result = convert_file(file_output_path, converter)

        if conversion_result.status == ConversionStatus.SUCCESS:
            converted_results_path = store_result_locally(conversion_result)
            delete_file(file_output_path)
            url = file.move_blob(PROCESSED_DOCUMENT_CONTAINER)

            metadata = {
                'converted': 'true',
                'original_file': url
            }

            if chunking_enabled:
                chunker = HybridChunker(
                    max_tokens=MAX_TOKENS,
                )
                chunks = chunker.chunk(dl_doc=conversion_result.document)
                docs = [Document(id=uuid.uuid1().hex, page_content=chunker.serialize(content), metadata=metadata) for content in chunks]
            else:
                docs = [Document(
                        id=uuid.uuid1().hex, 
                        page_content=conversion_result.document.export_to_markdown(), 
                        metadata={'file_url': url}
                    )]
                
            index_documents(search_index, docs)

            if upload_results:
                upload_container = Container(storage_url, DefaultAzureCredential(), PROCESSED_DOCUMENT_CONTAINER)
                upload_container.upload_from_local(converted_results_path, file.name.rsplit('.', 1)[0], metadata) 
        else:
            file.release_lease()
            print(f"Failed to convert {file.name}")

def create_search_index() -> AzureSearch:
    index_name: str = "document-index"

    return AzureSearch(
        azure_search_endpoint=os.getenv("AZURE_AI_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_AI_SEARCH_KEY"),
        index_name=index_name,
        embedding_function=embeddings_model.embed_query,
    )

def index_documents(search_index: AzureSearch, docs) -> None:
    search_index.add_texts(
        keys=[doc.id for doc in docs],
        texts=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs],
    )

def convert_file(path: Path, converter: DocumentConverter) -> ConversionResult:
    return converter.convert(path, raises_on_error=False)

def store_result_locally(result: ConversionResult) -> Path:
    if result.status == ConversionStatus.SUCCESS:
        dir = Path(OUTPUT_DIR / Path(result.document.origin.filename).stem)
        dir.mkdir(parents=True, exist_ok=True)
        md_filename = dir / f"result.md"
        result.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
        return dir
    else:
        print(f"Failed to convert {result.stem}")
        return None

def delete_file(file_name: str):
    os.remove(file_name)

if __name__ == "__main__":
    main()