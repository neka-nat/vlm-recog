# VLM-RECog

A Python library for zero-shot object detection and segmentation using Vision-Language Models (VLMs), built on Google's Gemini 2.5 Flash model.

## Installation

```bash
pip install vlm-recog
export GEMINI_API_KEY=<your_api_key>
```

### Requirements

- Python 3.11+
- Google Gemini API key (set as `GEMINI_API_KEY` environment variable)

## Quick Start

```python
from PIL import Image
from vlm_recog.detection import detect
from vlm_recog.visualization import draw_detections

# Load an image
image = Image.open("path/to/image.jpg")

# Detect objects by text labels
result = detect(image, ["dog", "bicycle", "person"])

# Visualize results
output_image = draw_detections(image, result)
output_image.show()
```

## Demo

![output](assets/output.png)
