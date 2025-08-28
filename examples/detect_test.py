from PIL import Image
from vlm_recog.detection import detect

from vlm_recog.visualization import draw_detections


image = Image.open("./examples/input.jpg")
result = detect(image, ["dog", "bicycle"])
output_image = draw_detections(image, result)
output_image.show()
