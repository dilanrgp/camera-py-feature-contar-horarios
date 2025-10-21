import threading
import datetime
from datetime import timezone
import hashlib
import requests
import base64
import cv2
import time
from face_detector import FaceDetector
from tracking import FaceTracker
from threading import Thread
import numpy as np
from sort import Sort
import os
import subprocess

device_mode = False
STREAM_TIME = 300
try:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate

    device_mode = True
except ImportError:
    print('Tensorflow library Import Error')
    device_mode = False


class CameraStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        # ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        ret = self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.stream.set(cv2.CAP_PROP_FPS, 30)

        # fps = int(self.stream.get(cv2.CAP_PROP_FPS))
        fps = 30

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
        if self.frame is None:
            os.system('sudo shutdown -r now')

        self.img_size = np.asarray(self.frame.shape)[0:2]
        self.fps = fps

        # Variable to control when the camera is stopped
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):      
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()
            if self.frame is None:
                os.system('sudo shutdown -r now')

    def read(self):
        # Return the most recent frame
        grabbed = self.grabbed
        self.grabbed = False
        copy_frame = self.frame.copy()

        # if not grabbed:
        #     print("Frame is not available")

        return grabbed, copy_frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, path):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(path)

        fps = int(self.stream.get(cv2.CAP_PROP_FPS))
        width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = fps
        self.img_size = np.asarray([height, width])

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        return self

    def read(self):
        (self.grabbed, self.frame) = self.stream.read()
        self.frame = cv2.resize(self.frame, (960, 540)) 
        self.img_size = np.asarray(self.frame.shape)[0:2]

        return self.grabbed, self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = [2, 5, 10, 17, 29, 40, 60, 75]
genderList = ['male', 'female']


class AgeGenderThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.sio = None
        self.config = None
        self.thread_running = True
        self.age_input_details = None
        self.age_output_details = None

        self.ageNet = None
        self.genderNet = None

        self.face_list = []
        self.start_stream_time = 0

        print('Age Gender Thread Init')
        self.initLibrary()
        self.lock = threading.Lock()

    def initLibrary(self):
        CWD_PATH = os.getcwd()
        age_model_path = 'model/age_model.tflite'
        if device_mode:
            self.age_interpreter = Interpreter(model_path=os.path.join(CWD_PATH, age_model_path))
            self.age_interpreter.allocate_tensors()

            self.age_input_details = self.age_interpreter.get_input_details()[0]
            self.age_output_details = self.age_interpreter.get_output_details()

            self.age_height = self.age_input_details['shape'][1]
            self.age_width = self.age_input_details['shape'][2]
        else:
            ageProto = "model/age_deploy.prototxt"
            ageModel = "model/age_net.caffemodel"
            genderProto = "model/gender_deploy.prototxt"
            genderModel = "model/gender_net.caffemodel"

            self.ageNet = cv2.dnn.readNet(ageModel, ageProto)
            self.genderNet = cv2.dnn.readNet(genderModel, genderProto)

    def predict_age_gender(self, frame):
        if device_mode:
            frame_resized = cv2.resize(frame, (self.age_width, self.age_height))
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(frame_resized.astype(np.float32), axis=0)

            self.age_interpreter.set_tensor(self.age_input_details['index'], input_data)
            st = time.time()
            self.age_interpreter.invoke()
            print("inference time:", time.time() - st)
            age_dist = self.age_interpreter.get_tensor(self.age_output_details[0]['index'])
            predicted_genders = self.age_interpreter.get_tensor(self.age_output_details[1]['index'])[0]
            ages = np.arange(0, 101).reshape(101, 1)
            predicted_ages = int(age_dist.dot(ages).flatten())
            gender = "male" if predicted_genders[0] < 0.5 else "female"
        else:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            self.genderNet.setInput(blob)
            genderPreds = self.genderNet.forward()
            gender = "male" if genderPreds[0].argmax() == 0 else "female"

            self.ageNet.setInput(blob)
            agePreds = self.ageNet.forward()
            predicted_ages = ageList[agePreds[0].argmax()]

        return predicted_ages, gender

    def run(self):
        print("Age Gender Thread Start")

        while self.thread_running:
            if len(self.face_list) > 0:
                face = self.face_list[0]
                age, gender = self.predict_age_gender(face.best_face)

                face.age = age
                face.gender = gender
                self.sendFaceOut(face)
                self.lock.acquire()
                self.face_list.remove(face)
                self.lock.release()
            else:
                time.sleep(1)

    def stop(self):
        self.thread_running = False

    def setThreadInfo(self, sio, config):
        self.sio = sio
        self.config = config
        print(config)

    def setStartStreamTime(self, start_stream_time):
        self.start_stream_time = start_stream_time

    def setFaceList(self, face_list):
        self.lock.acquire()
        self.face_list.extend(face_list)
        self.lock.release()

    def sendFaceOut(self, face):
        ts, keyhash = self.generateKeyHash()
        now = datetime.datetime.now()  # current date and time
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")

        if self.sio is not None:
            send_data = {
                'from': self.sio.sid,
                'to': 'ids.ladorian.' + str(self.config['idcamera']),
                'name' : 'face:out',
                'data' : {'id': str(face.face_id),}
            }

            if self.start_stream_time > 0 and time.time() - self.start_stream_time < STREAM_TIME:
                self.sio.emit('send:message', send_data)

        print("Face Out: ID=", face.face_id, ": Age=", face.age, "Gender=", face.gender)
        url = 'https://people.ladorianids.es/ws/people/create?timestamp=' + str(ts) + '&keyhash=' + keyhash
        age_type = ''
        if face.age < 15:
            age_type = 'child'
        elif face.age < 26:
            age_type = 'young'
        elif face.age < 66:
            age_type = 'adult'
        else:
            age_type = 'senior'

        duration = round(face.last_time - face.first_time, 3)
        data = {
            "customer_id": str(self.config['customer']['idcustomer']),
            "site_id": str(self.config['site']['idsite']),
            "camera_id": str(self.config['idcamera']),
            "email": self.config['email'],
            "gender": face.gender,
            "type": age_type,
            "age": face.age,
            "duration": str(duration),
            "positions": face.face_pos,
            "datetime": date_time
        }

        face_out_post_ok = False
        try:
            response = requests.put(url, json=data)
            if response.status_code == 200:
                face_out_post_ok = True
        except:
            print("Face Out post is error")

        if face_out_post_ok:
            if self.sio is not None:
                send_data = {
                    'from': self.sio.sid,
                    'to': 'ids.ladorian.camera',
                    'name': 'log:error',
                    'message': 'Face Out Posting is failed'
                }

                self.sio.emit('send:message', send_data)

        # print(response.content)

    def generateKeyHash(self):
        dt = datetime.datetime.now()
        utc_time = dt.replace(tzinfo=timezone.utc)
        ts = int(utc_time.timestamp() * 1000)
        key = str(ts) + "ALRIDKJCS1SYADSKJDFS"
        h = hashlib.sha256(key.encode('utf-8'))
        keyhash = h.hexdigest()

        return ts, keyhash


class CameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.sio = None
        self.config = None
        self.thread_running = True
        self.videostream = None
        self.detector = None
        self.tpu_detector = None
        self.tracker = FaceTracker()
        self.age_gender_proc = None
        print('Camera Thread Init')
        self.initLibrary()

    def initLibrary(self):
        model_path = 'model/detect_edgetpu.tflite'
        CWD_PATH = os.getcwd()

        if device_mode:
            self.tpu_detector = Interpreter(model_path=os.path.join(CWD_PATH, model_path),
                                            experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
            self.tpu_detector.allocate_tensors()
            # Get model details
            self.input_details = self.tpu_detector.get_input_details()[0]
            self.output_details = self.tpu_detector.get_output_details()

            self.tpu_height = self.input_details['shape'][1]
            self.tpu_width = self.input_details['shape'][2]
        else:
            self.detector = FaceDetector()

        self.videostream = CameraStream().start()
        self.age_gender_proc = AgeGenderThread()
        self.age_gender_proc.start()

        self.start_stream_time = 0

        time.sleep(1)

    def run(self):
        print("Camera Thread Start")

        def quality_assesment(box, original):
            h = box[3] - box[1]
            w = box[2] - box[0]
            wh = min(w, h)
            lower_threshold = 0.01
            upper_threshold = 0.1
            if (wh > upper_threshold):
                return original
            elif (wh < lower_threshold):
                return 0
            else:
                return original * (wh - lower_threshold) / (upper_threshold - lower_threshold)

        freq = cv2.getTickFrequency()
        # init tracker
        detect_interval = 1
        cnt = 0
        tracker = Sort(max_age=100)  # create instance of the SORT tracker
        colours = np.random.rand(32, 3)
        total_time = 0

        fpsLimit = 1000/self.videostream.fps # throttle limit
        startTime = time.time() * 1000

        while self.thread_running:
            nowTime = time.time() * 1000
            if int(nowTime - startTime) < fpsLimit:
                continue

            # Start timer (for calculating frame rate)
            t1 = cv2.getTickCount()
            # Grab frame from video stream
            grabbed, frame = self.videostream.read()
            if not grabbed or frame is None:
                if cv2.waitKey(20) == ord('q'):
                    break
            try:
                (imH, imW) = frame.shape[:2]
            except:
                if cv2.waitKey(20) == ord('q'):
                    break

            face_list = []
            st = time.time()
            if not device_mode:
                if cnt % detect_interval == 0:
                    face_list = self.detector.detect(frame)
            else:
                sx = 0
                sy = 0
                ex = imW
                ey = imH
                wid = ex - sx
                hei = ey - sy
                frame_resized = cv2.resize(frame[sy:ey, sx:ex], (self.tpu_width, self.tpu_height), cv2.INTER_AREA)
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                input_data = np.expand_dims(frame_resized, axis=0)
                self.tpu_detector.set_tensor(self.input_details['index'], input_data)
                self.tpu_detector.invoke()
                rects = self.tpu_detector.get_tensor(self.output_details[0]['index'])[0]
                confidences = self.tpu_detector.get_tensor(self.output_details[2]['index'])[0]

                for rc, confid in zip(rects, confidences):
                    if confid < 0.7:
                        break
                    margin = (rc[3] - rc[1]) * 0.2
                    x1 = int(max(-sx, (rc[1] - margin) * wid + sx))
                    y1 = int(max(-sy, rc[0] * hei) + sy)
                    x2 = int(min(imW, (rc[3] + margin) * wid + sx))
                    y2 = int(min(imH, rc[2] * hei + sy))
                    confid = quality_assesment(rc, confid)
                    face_list.append([x1, y1, x2, y2, confid])

            if face_list != []:
                face_list = np.array(face_list)

            # print(time.time() - st)

            cnt += 1

            trackers = tracker.update(face_list, self.videostream.img_size, detect_interval)

            current_face_list, leave_face_list, new_face_list = self.tracker.trackFace(frame, trackers)

            # send new face in event
            self.sendFaceIn(new_face_list)
            self.age_gender_proc.setFaceList(leave_face_list)

            for face in current_face_list:
                d = face.face_box
                cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colours[face.face_id % 32, :] * 255, 3)
                cv2.putText(frame, 'ID : %d' % (face.face_id), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, colours[face.face_id % 32 % 32, :] * 255, 2)

            # to draw the detection result
            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / freq
            total_time += time1
            frame_rate_calc = cnt / total_time

            cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

            email = ''
            if(self.config != None and self.config['email'] != None):
                email = self.config['email']

            title = 'Face detector | ' + email
            cv2.imshow(title, frame)
            # print("Frame Rate = ", frame_rate_calc)

            if self.start_stream_time > 0:
                if time.time() - self.start_stream_time < STREAM_TIME:
                    try:
                        self.p.stdin.write(frame.tostring())
                    except Exception as e:
                        print(e)
                else:   # stop stream
                    print('Steaming Stop Automatically')
                    self.start_stream_time = 0
                    self.age_gender_proc.setStartStreamTime(self.start_stream_time)
                    self.p.kill()

                    send_data = {
                        'from': self.sio.sid,
                        'to': 'ids.ladorian.' + str(self.config['idcamera']),
                        'name': 'streaming:closed',
                        'data': {}
                    }

                    if self.sio is not None:
                        self.sio.emit('send:message', send_data)



            startTime = time.time() * 1000 # reset time
            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Clean up
        cv2.destroyAllWindows()
        self.videostream.stop()
        self.age_gender_proc.stop()
        print("Camera Thread Stop")

    def setThreadInfo(self, sio, config):
        self.sio = sio
        self.config = config
        self.age_gender_proc.setThreadInfo(sio, config)

    def stop(self):
        self.thread_running = False
        self.age_gender_proc.stop()

    def sendFaceIn(self, face_list):
        if self.start_stream_time == 0:
            return

        for face in face_list:
            print("Face In: ID = ", str(face.face_id))
            #retval, buffer = cv2.imencode('.jpg', face.first_face)
            #jpg_as_text = base64.b64encode(buffer)

            send_data = {
                'from': self.sio.sid,
                'to': 'ids.ladorian.' + str(self.config['idcamera']),
                'name': 'face:in',
                'data': {
                    'id': str(face.face_id),
                    #'image': jpg_as_text,
                    'gender': "",
                    'age': "",
                }
            }

            #if self.sio is not None:
                #self.sio.emit('send:message', send_data)

    def startStream(self, rtmp_url):
        print('Steaming Start = ' + rtmp_url)

        if self.start_stream_time > 0:
            return

        command = ['ffmpeg',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(self.videostream.img_size[1], self.videostream.img_size[0]),
                        '-r', str(self.videostream.fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast',
                        '-f', 'flv',
                        rtmp_url]

        # start stream
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.start_stream_time = time.time()
        self.age_gender_proc.setStartStreamTime(self.start_stream_time)