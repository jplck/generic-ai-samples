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
from ..shared.storage import Container

dotenv.load_dotenv()

IMAGE_RESOLUTION_SCALE = 2.0
OUTPUT_DIR = Path("output")
DOCUMENT_CONTAINER = "documents"
PROCESSED_DOCUMENT_CONTAINER = 'processed-documents'

upload_results = os.getenv("UPLOAD_RESULTS", "false").lower() == "true"
storage_url = os.getenv("STORAGE_ACCOUNT_URL")

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

    container = Container(storage_url, DefaultAzureCredential(), DOCUMENT_CONTAINER)
    container.create_container()
    files = container.get_files()

    for file in files:
        file_output_path = Path(OUTPUT_DIR, file.name)

        if file.is_locked():
            continue

        file.lease()
        file.download(file_output_path)

        converted_results_path = convert_file(file_output_path, converter)

        if converted_results_path:
            delete_file(file_output_path)
            url = file.move_blob(PROCESSED_DOCUMENT_CONTAINER)

            #work with the results

            if upload_results:
                metadata = {
                    'converted': 'true',
                    'original_file': url
                }
                upload_container = Container(storage_url, DefaultAzureCredential(), PROCESSED_DOCUMENT_CONTAINER)
                upload_container.upload_from_local(converted_results_path, file.name.rsplit('.', 1)[0], metadata)
        else:
            file.release_lease()
            print(f"Failed to convert {file.name}")


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

def delete_file(file_name: str):
    os.remove(file_name)

if __name__ == "__main__":
    main()