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
from docling.datamodel.document import ConversionResult, ConversionStatus
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.blob import BlobLeaseClient
import os
import dotenv

dotenv.load_dotenv()

IMAGE_RESOLUTION_SCALE = 2.0
OUTPUT_DIR = Path("output")
DOCUMENT_CONTAINER = "documents"
PROCESSED_DOCUMENT_CONTAINER = 'processed-documents'

storage_url = os.getenv("STORAGE_ACCOUNT_URL")
default_credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url=storage_url, credential=default_credential)

@dataclass
class Blob:
    container: str
    blob_name: str

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

    dir = Path(OUTPUT_DIR)
    dir.mkdir(parents=True, exist_ok=True)

    create_container_if_not_exists(DOCUMENT_CONTAINER)

    files = get_list_of_files()
    for file in files:
        blob_client = blob_service_client.get_blob_client(file.container, file.blob_name)
        file_output_path = Path(OUTPUT_DIR, file.blob_name)

        if is_locked(blob_client):
            continue

        lease = create_lease(blob_client)
        download_blob(blob_client, file_output_path)

        converted_results_path = convert_file(file_output_path, converter)
        if converted_results_path:
            delete_file(file_output_path)
            url = move_blob(blob_client, lease.id, PROCESSED_DOCUMENT_CONTAINER)
            metadata = {
                'converted': 'true',
                'original_file': url
            }

            upload_folder(converted_results_path, PROCESSED_DOCUMENT_CONTAINER, file.blob_name.rsplit('.', 1)[0], metadata)
        else:
            lease.release()
            print(f"Failed to convert {file.blob_name}")

    
def create_container_if_not_exists(container_name: str) -> ContainerClient:
    document_container_client = blob_service_client.get_container_client(container_name)
    if not document_container_client.exists():
        document_container_client.create_container()
    return document_container_client

def is_locked(blob_client: BlobClient) -> bool:
    return blob_client.get_blob_properties().lease.status == 'locked'

def create_lease(blob_client: BlobClient) -> BlobLeaseClient:
    lease_client = BlobLeaseClient(blob_client)
    lease_client.acquire()
    return lease_client

def convert_file(path: Path, converter: DocumentConverter) -> Path:
    result = converter.convert(path, raises_on_error=False)
    return store_result_locally(result)

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

def get_list_of_files() -> list[Blob]:
    list = blob_service_client.get_container_client(DOCUMENT_CONTAINER).list_blobs()
    return [Blob(DOCUMENT_CONTAINER, blob.name) for blob in list]

def download_blob(blob_client: BlobClient, path: Path):
    blob_data = blob_client.download_blob().readall()
    with open(path, "wb") as f:
        f.write(blob_data)

def upload_folder(local_folder_path: str, container_name: str, folder_name: str, metadata: dict):
    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            blob_name = f"{folder_name}/{relative_path.replace('\\', '/')}"
            with open(local_file_path, "rb") as data:
                blob_service_client.get_blob_client(container_name, blob_name).upload_blob(data, overwrite=True, metadata=metadata)

def delete_file(file_name: str):
    os.remove(file_name)

def move_blob(blob_client: BlobClient, lease_id: str, container_name: str) -> Path:
    create_container_if_not_exists(container_name)
    target_blob_client = blob_service_client.get_blob_client(container_name, blob_client.blob_name)
    target_blob_client.start_copy_from_url(blob_client.url)
    blob_client.delete_blob(lease=lease_id)
    return target_blob_client.url

if __name__ == "__main__":
    main()