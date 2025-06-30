import logging
import tempfile
from typing import Any, Dict, List, Tuple
from PIL import Image
import easyocr
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """A class to extract text from images using EasyOCR with configurable settings."""

    def __init__(self):
        """
        Initialize the OCRProcessor with configuration.
        """
        self.lang = ["en"]
        self.max_size = 800
        
        try:
            self.reader = easyocr.Reader(self.lang)
            logger.info("EasyOCR initialized with languages: %s", self.lang)
        except Exception as e:
            logger.error("Failed to initialize EasyOCR: %s", str(e))
            raise

    def extract_text(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from an image and return list of dicts with text, bounding box and estimated color.
        """
        logger.info("Starting OCR for image: %s", image_path)
        resized_path = self._preprocess_image(image_path)
        original_image = Image.open(image_path).convert("RGB")

        try:
            result = self.reader.readtext(resized_path)
            scale = self._get_scale(image_path)
            processed_result = []

            for box, text, _ in result:
                scaled_box = [(int(x / scale), int(y / scale)) for x, y in box]
                region = self._crop_region(original_image, scaled_box)
                color = self._estimate_text_color(region)
                processed_result.append({
                    "text": text,
                    "box": scaled_box,
                    "color": color,
                })

            logger.info("OCR completed successfully for image: %s", image_path)
            return processed_result

        except Exception as e:
            logger.exception("OCR failed on image: %s", image_path)
            raise
        finally:
            import os
            if os.path.exists(resized_path):
                os.remove(resized_path)

    def _crop_region(self, image: Image.Image, box: List[Tuple[int, int]]) -> np.ndarray:
        """
        Crop a rectangular region from the image based on the given bounding box.
        """
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        left, top, right, bottom = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        cropped = image.crop((left, top, right, bottom))
        return np.array(cropped)

    def _estimate_text_color(self, region: np.ndarray) -> Tuple[int, int, int]:
        """
        Estimate the foreground (text) color using KMeans clustering on the region.
        """
        flat = region.reshape(-1, 3)
        if len(flat) < 2:
            return tuple(map(int, flat.mean(axis=0)))
        km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(flat)
        labels, counts = np.unique(km.labels_, return_counts=True)
        text_color_index = labels[np.argmin(counts)]
        return tuple(map(int, km.cluster_centers_[text_color_index]))
    
    def _preprocess_image(self, image_path: str) -> str:
        """
        Preprocess the image by resizing and saving to a temporary file.
        """
        try:
            resized_path = tempfile.mktemp(suffix=".png")
            with Image.open(image_path).convert("RGB") as im:
                scale = min(self.max_size / im.width, self.max_size / im.height, 1.0)
                if scale < 1.0:
                    new_size = (int(im.width * scale), int(im.height * scale))
                    try:
                        resample_filter = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample_filter = Image.LANCZOS
                    im = im.resize(new_size, resample=resample_filter)
                im.save(resized_path)
                logger.debug("Image resized and saved to: %s", resized_path)
            return resized_path
        except Exception as e:
            logger.error("Image preprocessing failed: %s", str(e))
            raise

    def _get_scale(self, image_path: str) -> float:
        """
        Calculate the scale factor used for resizing.
        """
        with Image.open(image_path) as im:
            return min(self.max_size / im.width, self.max_size / im.height, 1.0)
    
