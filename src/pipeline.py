import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
from PIL import Image
from processors.extract_texts import OCRProcessor
from processors.text_remover import TextRemoverGenerator
from processors.png2svg import ImageCleaner, SVGVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    """
    A processing pipeline to:
    1. Extract text regions from an input image using OCR.
    2. Generate a cleaned image with text removed via an external service.
    3. Convert the cleaned image and extracted text metadata into an SVG vector output.
    """

    def __init__(self, api_key: str, input_path: str, output_path: str):
        """
        Initialize the pipeline with input/output paths.
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_dir = self.output_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ocr_processor = OCRProcessor()
        self.text_remover = TextRemoverGenerator(api_key=api_key)
        self.svg_converter = SVGVectorizer()

    def run(self):
        """
        Execute the full pipeline: OCR → inpainting → cleaning → vectorization.
        """
        logging.info("Running OCR...")
        ocr_result = self.ocr_processor.extract_text(self.input_path)

        openai_output_path = self.text_remover.generate_image(str(self.input_path))
        if openai_output_path is None:
            raise RuntimeError("Text removal failed. No image was generated.")

        with Image.open(self.input_path) as original_img:
            original_size = original_img.size

        with Image.open(openai_output_path) as generated_img:
            if generated_img.size != original_size:
                resized_img = generated_img.resize(original_size)
                resized_img.save(openai_output_path)
                logging.info(f"Resized OpenAI output image to match original size: {original_size}")

        cleaner = ImageCleaner(openai_output_path)
        clean_image_path, texts = cleaner.clean_and_extract_texts(ocr_result)

        logging.info("Vectorizing image...")
        self.svg_converter.convert(clean_image_path, str(self.output_path), texts)

        os.remove(clean_image_path)


if __name__ == "__main__":
    load_dotenv()
    
    if len(sys.argv) != 3:
        print("Usage: python pipeline.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be defined in .env")

    pipeline = Pipeline(api_key, input_path, output_path)
    pipeline.run()
