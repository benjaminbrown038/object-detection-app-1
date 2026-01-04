import cv2
import torch
from models.tiny_yolo import TinyYOLO

model = TinyYOLO()
model.load_state_dict(torch.load("tiny_yolo.pt"))
model.eval()

img = cv2.imread("test.jpg")
img_r = cv2.resize(img, (224, 224))
x = torch.from_numpy(img_r).permute(2,0,1).float().unsqueeze(0)/255

with torch.no_grad():
    pred = model(x).mean(dim=[2,3])[0]

cx, cy, w, h, conf, _ = pred
H, W, _ = img.shape

x1 = int((cx - w/2) * W)
y1 = int((cy - h/2) * H)
x2 = int((cx + w/2) * W)
y2 = int((cy + h/2) * H)

cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
cv2.imshow("Detection", img)
cv2.waitKey(0)
