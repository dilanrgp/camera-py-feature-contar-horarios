import cv2


def quality_assesment(box):
    original = box[0]
    h = box[4] - box[2]
    w = box[3] - box[1]
    wh = min(w, h)
    lower_threshold = 0.05
    upper_threshold = 0.2
    if (wh > upper_threshold):
        return original
    elif (wh < lower_threshold):
        return 0
    else:
        return original * (wh - lower_threshold) / (upper_threshold - lower_threshold)


class FaceDetector:
    def __init__(self):
        # load opencv dnn face detector
        modelFile = "model/face_detector.pb"
        configFile = "model/face_detector.pbtxt"
        self.model = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    def detect(self, image, conf_threshold=0.5):
        frameHeight = image.shape[0]
        frameWidth = image.shape[1]
        # convert image to blob, mean subtraction and scaling
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)

        self.model.setInput(blob)
        detections = self.model.forward()
        rect = []
        if detections.shape[2] == 0:
            return rect

        confidence = detections[0, 0, 0, 2]
        if confidence < conf_threshold:
            return rect

        # convert detection result to dlib rectangle to use in 2d landmark detection
        for det in detections[0, 0]:
            if det[2] < conf_threshold:  # confidence
                break
            margin = (det[5] - det[3]) * 0.1
            x1 = int(max(0, det[3] - margin) * frameWidth)  # left
            y1 = int(max(0, det[4]) * frameHeight)  # top
            x2 = int(min(1, det[5] + margin) * frameWidth)  # right
            y2 = int(min(1, det[6]) * frameHeight)  # bottom
            quality = quality_assesment(det[2:])
            rect.append([x1, y1, x2, y2, quality])
        return rect
