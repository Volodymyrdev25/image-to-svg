import os
import base64
from openai import OpenAI
from typing import Optional
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextRemoverGenerator:
    """
    A class to generate modified images for coloring books using the OpenAI Vision API.
    This generator removes text from an input image and saves the cleaned output image.
    """

    def __init__(self, api_key: str, media_dir: str = "./media"):
        """
        Initialize the generator with the OpenAI API key and media directory.
        """
        self.client = OpenAI(api_key=api_key)
        self.media_dir = media_dir
        self.output_dir = os.path.join(self.media_dir, "openai_outputs")
        os.makedirs(self.output_dir, exist_ok=True)

    def _create_file(self, file_path: str) -> str:
        """
        Uploads the input file to OpenAI and returns the file ID.
        """
        with open(file_path, "rb") as f:
            result = self.client.files.create(file=f, purpose="vision")
        return result.id
    
    def generate_image(self, input_image_path: str) -> Optional[str]:
        """
        Sends the image to OpenAI to remove all text content and save the output.
        """
        try:
            file_id = self._create_file(input_image_path)
            
            prompt = (
                "Remove all visible text from the image while keeping everything else "
                "(including background, colors, textures, and resolution) exactly the "
                "same. Do not alter image size or any other visual detail."
            )  
            
            response = self.client.responses.create(
                model="gpt-4.1",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "file_id": file_id},
                        ],
                    }
                ],
                tools=[{"type": "image_generation"}],
            )

            image_generation_calls = [
                output
                for output in response.output
                if output.type == "image_generation_call"
            ]

            if not image_generation_calls:
                logger.warning("No image generation result.")
                return None

            image_base64 = image_generation_calls[0].result
            output_path = os.path.join(self.output_dir, f"open_ai_output.png")

            with open(output_path, "wb") as f:
                f.write(base64.b64decode(image_base64))

            return output_path

        except Exception as e:
            logger.exception(f"Image generation failed: {e}")
            return None
