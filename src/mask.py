import cv2
import numpy as np

drawing = False
ix, iy, ex, ey = -1, -1, -1, -1
img = cv2.imread('../data/unit_test/img1.jpg')
clone = img.copy()

result = None  # Initialize result to avoid NameError if 'q' is pressed first

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = clone.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        cv2.rectangle(img, (ix, iy), (ex, ey), (0, 255, 0), 2)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_rectangle)

while True:
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('g') and ex != -1:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (min(ix, ex), min(iy, ey), abs(ex - ix), abs(ey - iy))
        cv2.grabCut(clone, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        result = clone * mask2[:, :, np.newaxis]
        cv2.imshow('segmented', result)
    elif k == ord('q'):
        if result is not None:
            cv2.imwrite('../data/unit_test/grabcut_mask.png', result.copy())
        break

cv2.destroyAllWindows()