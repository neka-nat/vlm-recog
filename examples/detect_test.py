from PIL import Image
from vlm_recog.detection import detect

from vlm_recog.visualization import draw_detections
from vlm_recog.bbox_utils import merge_boxes


image = Image.open("./examples/b9-1.png")
result = detect(image, ["正面図", "背面図", "円", "寸法線"])
print([r.box_2d for r in result])
bboxs = merge_boxes([r.box_2d for r in result])
print(bboxs)
output_image = draw_detections(image, result)
output_image.show()
