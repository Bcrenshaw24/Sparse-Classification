import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
from sparse import edge_det
model = YOLO("yolov8n.pt")

def detect(img):
    results = model(img)
    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box[:4])

        cropped_object = img[y1:y2, x1:x2]
        
        return cropped_object
        
def mask(img): 
    clone = img.copy()
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    height, width, channels = img.shape
    # Inset the rectangle by 1 pixel to allow background sampling at the edges
    rect = (1, 1, width - 2, height - 2)

    cv2.grabCut(clone, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    return clone * mask2[:, :, np.newaxis]


def process(src, train=True): 
    dir_path = Path(src)
    i = 0
    for item in dir_path.iterdir():
        i += 1
        folder = "../data/train/" if train else "../data/test/"
        full = folder + item.name
        img = cv2.imread(full)
        masked = mask(img)
        edges = edge_det("sobel", masked)
        cv2.imwrite(f"../data/post_seg_train/{item.name}", edges)

process("../data/train")