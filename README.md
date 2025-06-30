# SVG Converter

This project converts a PNG image into an SVG file by:

- Extracting text using OCR (EasyOCR)
- Removing the text using OpenAI inpainting
- Vectorizing the background image
- Re-inserting the text as `<text>` elements in the SVG

## Requirements

- Docker & Docker Compose
- OpenAI API Key

## Setup

1. Create a `.env` file with the following content:
   `OPENAI_API_KEY=your_openai_key`

2. Make the start script executable:
   `chmod +x start`

3. Build the Docker image (only needed once):
   `docker compose build svg_converter`

## Usage

- To convert an image:
  `./start ./media/inputs/your_image_name.png ./media/outputs/output.svg`

### Notes

- Input images should be placed in the inputs/ directory.
- Output SVG files will be saved in the outputs/ directory.
- The container uses EasyOCR, OpenAI API, and scikit-image to process and vectorize the image.
