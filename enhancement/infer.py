import time
import cv2
import torch
import imutils
from imutils.video import VideoStream

from .model import EnhanceCNN


def predict(model, webcam=True, image=None):
    if webcam:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            val = model.predict(frame, webcam)
            cv2.imshow("Frame", val)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    else:
        if image:
            image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            val = model.predict(image, False)
            cv2.imshow("Output", val)
            cv2.waitKey(0)
            cv2.destroyWindow("Output")


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
model = EnhanceCNN().to(device)

# For webcam feed
# predict(model)

# For file from local disk
# predict(model, False, "path to file")
