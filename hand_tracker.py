import cv2
import numpy as np
import onnxruntime as ort
import math
from config import PINCH_THRESHOLD, WIDTH, HEIGHT

class HandTracker:
    def __init__(self):
        # Use an ONNX version of the hand landmarker model
        # You will need to download 'hand_landmark.onnx' to your project folder
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession("hand_landmark.onnx", providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def process(self, frame):
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, :]

        outputs = self.session.run(None, {self.input_name: img})
    
        # Reshape the flat 63-value array into (21, 3) 
        # to get [x, y, z] for each of the 21 points
        landmarks = outputs[0].flatten().reshape(21, 3) 
    
        data = self._extract(landmarks, w, h)
        return {"left": None, "right": data}

    def _extract(self, landmarks, w, h):
        # landmarks is now (21, 3), so lm[0] and lm[1] will work
        pts = [(lm[0], lm[1]) for lm in landmarks]
        px_coords = [(int(x * w), int(y * h)) for x, y in pts]
    
        # Calculate palm center using the first 5 landmarks (wrist and MCPs)
        palm_x = sum(p[0] for p in pts[:5]) / 5
        palm_y = sum(p[1] for p in pts[:5]) / 5
    
        return {
            "palm_norm": (palm_x, palm_y),
            "palm_px": (int(palm_x * w), int(palm_y * h)),
            "finger_count": self._count_fingers(pts),
            "is_pinched": math.dist(pts[4], pts[8]) < PINCH_THRESHOLD,
            "index_tip_px": px_coords[8],
        }
    def _count_fingers(self, pts):
        # Simplified logic: tip Y < pip Y
        count = 0
        if pts[8][1] < pts[6][1]: count += 1 # Index
        if pts[12][1] < pts[10][1]: count += 1 # Middle
        return count
