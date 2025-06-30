from __future__ import annotations
import logging, tempfile
from typing import Any, List, Tuple, Dict
import numpy as np
from PIL import Image
import svgwrite
from skimage import measure
from sklearn.cluster import KMeans
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageCleaner:
    """
    Cleans image text annotations and prepares metadata for SVG rendering.
    """

    def __init__(self, image_path: str) -> None:
        """
        Load the image from the given path into a NumPy array.
        
        """
        self.image_path = image_path
        self.arr = np.array(Image.open(image_path).convert("RGB"))
        logger.info("Image loaded from %s", image_path)

    def clean_and_extract_texts(self, ocr_result: List[Dict[str, Any]]) -> Tuple[str, List[Dict]]:
        """
        Prepares metadata for SVG text rendering based on OCR output,
        and saves a temporary cleaned version of the image.
        """
        texts_meta: List[Dict[str, Any]] = []
        logger.info("Extracting text metadata from OCR results")

        for item in ocr_result:
            text = item["text"]
            box = item["box"]
            fill = item["color"]

            xs, ys = zip(*box)
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            pad = 5
            region = self.arr[max(y0 - pad, 0):y1 + pad, max(x0 - pad, 0):x1 + pad]
            if region.size == 0:
                continue

            size = max(1, int((y1 - y0) * 0.9))

            texts_meta.append(
                dict(
                    text=text,
                    pos=(x0, y1),
                    size=size,
                    fill=f"rgb{fill}",
                    alignment_baseline="alphabetic",
                    text_anchor="start",
                )
            )

        tmp_path = tempfile.mktemp(suffix=".png")
        Image.fromarray(self.arr).save(tmp_path)
        logger.info("Temporary cleaned image written to %s", tmp_path)

        return tmp_path, texts_meta


class SVGVectorizer:
    """
    Converts a cleaned image and associated text metadata to an SVG file.
    """

    def __init__(self, n_colors: int = 128, min_tol: float = 0.2, max_tol: float = 1.5) -> None:
        """
        Initialize the vectorizer with quantization and contour simplification parameters.

        Args:
            n_colors: Number of color clusters to reduce the image to.
            min_tol: Minimum tolerance for contour simplification.
            max_tol: Maximum tolerance for contour simplification.
        """
        self.n_colors = n_colors
        self.min_tol = min_tol
        self.max_tol = max_tol

    def convert(self, img_path: str, svg_path: str, texts: List[Dict[str, str | Tuple[int, int] | int]]) -> None:
        """
        Converts a raster image and associated texts into an SVG.
        """
        logger.info("Loading image for vectorization: %s", img_path)
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        h, w, _ = arr.shape

        logger.info("Quantizing image into %d colors", self.n_colors)
        palette_img, labels, palette = self._quantize(arr)
        bg = self._most_frequent_color(arr)

        logger.info("Creating SVG drawing: %s", svg_path)
        dwg = svgwrite.Drawing(svg_path, size=(w, h))
        bg_rgb = tuple(int(x) for x in bg)
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), fill=f"rgb{bg_rgb}"))

        logger.info("Extracting contours from color regions")
        contours = self._extract_contours(labels, palette, bg)

        max_perimeter = max((p for _, _, p in contours), default=1.0)

        for col, cnt, perimeter in contours:
            tol = min(
                max(self.min_tol, self.min_tol + (self.max_tol - self.min_tol) * perimeter / max_perimeter),
                self.max_tol
            )
            simp = measure.approximate_polygon(cnt, tol)
            if len(simp) >= 3:
                d = "M " + " L ".join(f"{x:.3f},{y:.3f}" for y, x in simp) + " Z"
                fill_col = tuple(int(x) for x in col)
                dwg.add(dwg.path(d=d, fill=f"rgb{fill_col}", stroke="none"))

        logger.info("Adding OCR texts to SVG")
        for t in texts:
            dwg.add(
                dwg.text(
                    t["text"],
                    insert=t["pos"],
                    font_size=f'{t["size"]}px',
                    font_family="Arial",
                    fill=t["fill"],
                    alignment_baseline=t["alignment_baseline"],
                    text_anchor=t["text_anchor"],
                )
            )

        dwg.save()
        logger.info("SVG saved successfully to %s", svg_path)

    def _most_frequent_color(self, arr: np.ndarray) -> Tuple[int, int, int]:
        """
        Determine the most common RGB color in the image.
        """
        pixels = arr.reshape(-1, 3)
        counts = Counter(map(tuple, pixels))
        return counts.most_common(1)[0][0]

    def _quantize(self, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduce the number of colors in the image using KMeans clustering.
        """
        h, w, c = arr.shape
        flat = arr.reshape(-1, c)
        km = KMeans(n_clusters=self.n_colors, n_init=3, random_state=0).fit(flat)
        labels = km.labels_.reshape(h, w)
        centers = km.cluster_centers_.astype(np.uint8)
        return centers[labels], labels, centers

    def _extract_contours(
        self, lbl: np.ndarray, palette: np.ndarray, bg: Tuple[int, int, int]
    ) -> List[Tuple[Tuple[int, int, int], np.ndarray, float]]:
        """
        Extract polygon contours from non-background regions.
        """
        contours = []
        for idx, col in enumerate(palette):
            if tuple(col) == bg:
                continue
            mask = (lbl == idx).astype(np.uint8)
            cnts = measure.find_contours(np.pad(mask, 1), 0.01)
            cnts = [c - 1 for c in cnts if len(c) > 2]
            for c in cnts:
                p = np.sum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))
                contours.append((tuple(map(int, col)), c, p))
        return contours
