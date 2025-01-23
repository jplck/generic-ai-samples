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

IMAGE_RESOLUTION_SCALE = 2.0
OUTPUT_DIR = Path("output")

@dataclass
class Document:
    text: str
    original_file: str
    processed_file_folder: str
    chunks: list[str]

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

    convert_files("./tmp", converter)

def convert_files(directory: str, converter: DocumentConverter):
    files = get_list_of_files(directory)
    results = converter.convert_all(files, raises_on_error=False)
    store_results(results)

def store_results(results: Iterable[ConversionResult]):
    for res in results:
        if res.status == ConversionStatus.SUCCESS:
            dir = Path(OUTPUT_DIR / Path(res.document.origin.filename).stem)
            dir.mkdir(parents=True, exist_ok=True)
            md_filename = dir / f"result.md"
            res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)
        else:
            print(f"Failed to convert {res.stem}")

def get_list_of_files(directory: str) -> list[Path]:
    return [f for f in Path(directory).rglob("*.pdf")]

if __name__ == "__main__":
    main()