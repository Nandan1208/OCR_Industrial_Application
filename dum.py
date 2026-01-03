import cv2
from pyzbar.pyzbar import decode

img = cv2.imread("barcodee/n.png", cv2.IMREAD_GRAYSCALE)

# optional (but safe)
# _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

results = decode(img)

print("Detected:", len(results))
for r in results:
    print("TYPE:", r.type)
    print("DATA:", r.data.decode("utf-8"))
