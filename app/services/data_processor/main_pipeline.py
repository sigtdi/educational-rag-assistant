import json
import torch
import gc
from pathlib import Path

from processing.marker_processor import MarkerProcessor
from processing.text_processor import TextProcessor
from processing.image_processor import ImageProcessor
from processing.formula_processor import FormulaProcessor
from app.logger_setup import log

class PDFProcessingPipeline:
    def __init__(self):
        self.marker_process = MarkerProcessor()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()
        self.formula_processor = FormulaProcessor()

    def run(self, pdf_path: str, document_name: str = '', document_author: str = ''):
        chunks = self.marker_process.run_marker(pdf_path)
        chunks = self.text_processor.process(chunks, pdf_path)
        chunks = self.formula_processor.process(chunks, pdf_path)
        chunks = self.image_processor.process(chunks, pdf_path)


        self.output_folder = Path(__file__).resolve().parent / 'output'
        self.output_folder.mkdir(exist_ok=True, parents=True)

        print(chunks)

if __name__ == "__main__":
    parser = PDFProcessingPipeline()
    parser.run("C:/Users/Yana/Downloads/Alg-graphs-full_organized_removed.pdf")
    print("🎉 Parsing completed!")