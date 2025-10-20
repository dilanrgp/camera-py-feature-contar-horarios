import threading
import cv2
import time
import numpy as np
FACE_LEAVE_TIME = 5


class FaceTrajectory():
    def __init__(self, face, face_id, face_box, face_conf, face_embedding,gender_insightface,age_insightface):        
        self.face_box = face_box
        self.first_face = face
        self.last_face = face
        self.best_face = face
        self.face_id = face_id
        self.gender = None
        self.age = None
        self.face_conf = face_conf
        # insightface
        self.face_embedding = face_embedding
        self.gender_insightface = gender_insightface
        self.age_insightface = age_insightface

        self.first_time = time.time()
        self.last_time = time.time()

        (startX, startY, endX, endY) = face_box
        self.face_pos = [
                            {
                                "x": (startX + endX) / 2,
                                "y": (startY + endY) / 2
                            }
                         ]
        self.last_pos_time = time.time()

        # cv2.imshow(str(face_id), face)
        # cv2.imwrite("image/" + str(face_id) + ".bmp", face)

    def pushFaceInfo(self, face, face_box, face_conf):
        self.face_box = face_box
        self.last_face = face
        self.last_time = time.time()
        # cv2.imshow('last_face' + str(self.face_id), face)
        if face_conf > self.face_conf:
            self.face_conf = face_conf
            self.best_face = face
            # cv2.imshow(str(self.face_id), face)

        now = time.time()
        if now - self.last_pos_time > 0.8:
            (startX, startY, endX, endY) = face_box
            self.face_pos.append(
                {
                    "x": (startX + endX) / 2,
                    "y": (startY + endY) / 2
                }
            )

            self.last_pos_time = now


class FaceTracker():

    def __init__(self):
        print("Face trakcer init")
        self.face_list = []


    def trackFace(self, frame, trackers):
        new_face_list = []
        leave_face_list = []
        current_face_list = []

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        # get old face list
        cur_time = time.time()
        for row in self.face_list:
            if row.last_time + FACE_LEAVE_TIME < cur_time:
                leave_face_list.append(row)
                self.face_list.remove(row)
                print("Leave Face = " + str(row.face_id))

        num = 0
        for d in trackers:
            face_conf = d[5]
            d = d.astype(np.int32)
            (startX, startY, endX, endY) = (d[0], d[1], d[2], d[3])
            w = endX - startX
            h = endY - startY
            diff = int(0.3 * w)
            x1 = max(0, startX - diff)
            x2 = min(frameWidth, endX + diff)
            diff = int(0.3 * h)
            y1 = max(0, startY - diff)
            diff = int(0.3 * h)
            y2 = min(frameHeight, endY + diff)
            face = frame[y1:y2, x1:x2]            
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20 \
                    or fW > 1.4 * fH or 1.4 * fW < fH \
                    or fW * fH * 6 > frameWidth * frameHeight:  # not tracked and square
                continue

            # cv2.imshow("Face", face)

            tracked_id = 0
            selected_face = None

            # check exist face id
            for row in self.face_list:
                if row.face_id == d[4]:
                    selected_face = row
                    tracked_id = d[4]
                    break


            # generate new trajectory
            facebox = (startX, startY, endX, endY)

            if tracked_id == 0:
                selected_face = FaceTrajectory(face, d[4], facebox, face_conf)
                new_face_list.append(selected_face)
            else:
                selected_face.pushFaceInfo(face, facebox, face_conf)

            current_face_list.append(selected_face)

            num += 1

        self.face_list.extend(new_face_list)

        return current_face_list, leave_face_list, new_face_list
