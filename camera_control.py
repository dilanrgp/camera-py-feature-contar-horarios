import asyncio
import base64
import datetime
import hashlib
import math
import os
import sched
import subprocess
import threading
import time
from datetime import timezone
from random import randint
from threading import Thread
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
from device import Device

try:
    import websockets
except ImportError:  # pragma: no cover - optional dependency for websocket input
    websockets = None

STREAM_TIME = 86400
import requests
from insightface.app import FaceAnalysis

from distance import findCosineskLearn
from utils import getQualityFilters
from face_store import FaceStore

STREAM_TIME = 300
video = 0
#delete_faces = []
class CameraStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self,width=640,height=480):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(video)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        # fps = int(self.stream.get(cv2.CAP_PROP_FPS))
        fps = 30
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()
        if self.frame is None:
            print("Not frame")
            #Device.reboot()
            self.img_size = np.asarray([height, width])
        else:
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
                print("Not frame")
                #Device.reboot()

    def read(self):
        # Return the most recent frame
        grabbed = self.grabbed
        self.grabbed = False
        copy_frame = self.frame.copy()
        if self.frame is None:
            return False, None
        return grabbed, copy_frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

class WebSocketCameraStream:
    """Video stream that receives JPEG frames over a websocket connection."""

    def __init__(self, url, width=640, height=480):
        if websockets is None:
            raise ImportError('websockets package is required for ws:// video sources')

        parsed = urlparse(url)
        if parsed.scheme not in ("ws", "wss"):
            raise ValueError(f"Unsupported websocket URL: {url}")

        self.host = parsed.hostname or "0.0.0.0"
        self.port = parsed.port or 8765
        self.width = width
        self.height = height
        self.frame = None
        self.grabbed = False
        self.lock = threading.Lock()
        self.loop = None
        self.server = None
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.fps = 30
        self.ready = threading.Event()

    def start(self):
        self.thread.start()
        # Espera a que el servidor quede listo para aceptar conexiones.
        self.ready.wait(timeout=5)
        return self

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        start_server = websockets.serve(self._handler, self.host, self.port, max_size=None)
        self.server = self.loop.run_until_complete(start_server)
        self.ready.set()
        try:
            self.loop.run_forever()
        finally:
            if self.server is not None:
                self.server.close()
                self.loop.run_until_complete(self.server.wait_closed())
            self.loop.close()

    async def _handler(self, websocket):
        async for message in websocket:
            if isinstance(message, (bytes, bytearray)):
                frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    if self.width and self.height:
                        frame = cv2.resize(frame, (self.width, self.height))
                    with self.lock:
                        self.frame = frame
                        self.grabbed = True
            await asyncio.sleep(0)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            frame_copy = self.frame.copy()
            grabbed = self.grabbed
            self.grabbed = False
        return grabbed, frame_copy

    def stop(self):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join(timeout=2)

ageList = [2, 5, 10, 17, 29, 40, 60, 75]
genderList = ['male', 'female']

class AgeGenderThread(threading.Thread):
    '''
    Este thread es que se encarga de enviar los datos de las caras detectadas
    Para ello se ejecuta todo el rato y cada n segundos recoge el pandas dataframe y envía
    la información de las caras al webservice
    '''
    def __init__(self,new_people_time):
        threading.Thread.__init__(self)
        self.sio = None
        self.config = None
        self.thread_running = True
        self.start_stream_time = 0
        # útil para debugar y no tener que esperar 
        #new_people_time = 15
        self.time_send_faces = new_people_time
        print('Age Gender Thread Init')
        self.lock = threading.Lock()

    def run(self):
        global all_embeddings, face_store
        initial_time = 0
        print("Age Gender Thread Start")
        start_time = time.time()
        face_store = {}
        while self.thread_running:
            # bucle esperando los n segundos para enviar la info al webservice
            time.sleep(2)
            if not face_store:
                print("Empty people dataframe")
            else:
                #for face in face_store:
                for key,face in face_store.data.copy().items():
                    time_face = int(time.time() - face.last_time)
                    if time_face > self.time_send_faces :
                        sendedFaceOut = self.sendFaceOut(face)
                        print(sendedFaceOut)
                        if(sendedFaceOut):
                            face_store.remove(face.id)
                        #  remove all embedings from the deleted face
                        for i in all_embeddings.copy():
                            if i[0] == str(face.id):
                                all_embeddings.remove(i)
                #     #self.lock.acquire()
                #     #self.face_list.remove(face)
                #     #self.lock.release()



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
        duration = int(face.last_time - face.first_time)
        date_time = datetime.datetime.fromtimestamp(face.first_time).strftime("%Y-%m-%d %H:%M:%S")

        if self.sio is not None:
            send_data = {
                'from': self.sio.sid,
                'to': 'ids.ladorian.camera.' + str(self.config['idcamera']),
                'name' : 'face:out',
                'data' : {'id': str(face.id),}
            }
            if self.start_stream_time > 0 and time.time() - self.start_stream_time < STREAM_TIME:
                self.sio.emit('send:message', send_data)
            
        age_type = ''
        if face.age < 15:
            age_type = 'child'
        elif face.age < 36:
            age_type = 'young'
        elif face.age < 66:
            age_type = 'adult'
        else:
            age_type = 'senior'

        gender_str = 'male'
        if face.gender == 'F':
            gender_str = 'female'
                
        print("Face Out: ID=", face.id, ": Age=", age_type, "Gender=", face.gender, "Duration=", duration, "Camera=", self.config['email'])
        url = 'https://people.ladorianids.es/ws/people/create?timestamp=' + str(ts) + '&keyhash=' + keyhash
        #url = 'http://localhost:3000/ws/people/create?timestamp=' + str(ts) + '&keyhash=' + keyhash

        data = {
            "customer_id": int(self.config['customer']['idcustomer']),
            "site_id": int(self.config['site']['idsite']),
            "camera_id": int(self.config['idcamera']),
            "email": self.config['email'],
            "gender": gender_str,
            "type": age_type,
            "age": face.age,
            "duration": int(duration),
            "positions": face.pos,
            "datetime": date_time
            # embedding:
        }
        face_out_post_ok = False
        try:
            print(url)
            response = requests.put(url, json=data)
            if response.status_code == 200:
                face_out_post_ok = True
                print("Response= ", response)
            else:
                print("Error= ", response)
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
        return face_out_post_ok;
        # print(response.content)

    def getTimestamp(self): 
        dt = datetime.datetime.now()
        utc_time = dt.replace(tzinfo=timezone.utc)
        ts = int(utc_time.timestamp() * 1000)
        return ts

    def generateKeyHash(self):
        ts = self.getTimestamp()
        key = str(ts) + "ALRIDKJCS1SYADSKJDFS"
        h = hashlib.sha256(key.encode('utf-8'))
        keyhash = h.hexdigest()
        return ts, keyhash


'''
Este es el Thread que se encarga de las detecciones de caras
Recogiendo los frames que le llegan de otro thread que 
le la webcama
'''
class CameraThread(threading.Thread):
    def __init__(self,config,data_detector):
        threading.Thread.__init__(self)
        # variables globales para intercambiar información entre threads
        global all_embeddings
        global video
        all_embeddings = []
        self.sio = None
        self.config = config
        self.thread_running = True
        self.videostream = None
        self.detector = None
        self.tpu_detector = None
        #self.tracker = FaceTracker()
        self.age_gender_proc = None
        self.Ids = 1
        self.data_detector = data_detector
        self.Y_start, self.Y_end, self.X_start, self.X_end = 0, 0, 0, 0
        self.tiempo_estancia_cliente = 0
        self.video_source = self.data_detector.get('video', 0)
        self.start_stream_time = 0
        self.initLibrary()
        self.configs = {}
        self.rtmp_key = ''

    def initLibrary(self):
        self.detector = FaceAnalysis(name="buffalo_s",allowed_modules=['detection', 'genderage', 'recognition'])
        self.detector.prepare(ctx_id=0, det_size=(640, 640))
        if (self.config != None and self.config.get('section_start_y') != None):
            self.Y_start = self.config['section_start_y']
            section_height = self.config['section_height']
            self.Y_end = self.Y_start + section_height
            self.X_start = self.config['section_start_x']
            section_width  = self.config['section_width']
            self.X_end = self.X_start + section_width
        else:
            self.Y_start = self.data_detector['Y_start']
            self.Y_end = self.data_detector['Y_end']
            self.X_start = self.data_detector['X_start']
            self.X_end = self.data_detector['X_end']
        if (self.config != None and self.config.get('face_timeout') != None):
            self.tiempo_estancia_clientes = self.config['face_timeout']
        else:
            self.tiempo_estancia_clientes =  self.data_detector['tiempo_estancia_clientes']
        #self.tiempo_estancia_clientes =  15
        self.width = self.X_end - self.X_start
        self.height = self.Y_end - self.Y_start
        #Paso el tamaño de la detección a la cámara para que nos envíe frames de ese tamaño/posición
        self.videostream = self._create_video_stream().start()
        self.start_stream_time = 0
        self.age_gender_proc = AgeGenderThread(self.tiempo_estancia_clientes)
        self.age_gender_proc.start()
        time.sleep(1)

    def _create_video_stream(self):
        source = self.video_source
        if isinstance(source, str):
            parsed = urlparse(source)
            if parsed.scheme in ("ws", "wss"):
                width = self.width if self.width > 0 else 640
                height = self.height if self.height > 0 else 480
                return WebSocketCameraStream(source, width=width, height=height)
            try:
                numeric = int(source)
                source = numeric
            except (ValueError, TypeError):
                pass
        global video
        video = source
        width = self.width if self.width > 0 else 640
        height = self.height if self.height > 0 else 480
        return CameraStream(width=width, height=height)


    def run(self):
        global all_embeddings, face_store
        print("Camera Thread Start")
        email = self.config['email']
        face_store = FaceStore()
        face_store.setThreadInfo(self.sio, self.config)
        self.configs  = {}
        freq = cv2.getTickFrequency()
        detect_interval = 1
        #fpsLimit = 1000/self.videostream.fps # throttle limit
        cnt, new_person, total_time, ID = 0, 0, 0, 0
        startTime = time.time() * 1000
        # las variables de config pueden venir del webservice (data_detector) o del archivo de configuración (self.config)
        self.configs  = getQualityFilters(self.config, self.data_detector) 
        inSchedule = True
        while self.thread_running:
            if len(self.config['powers']):
                weekday = datetime.datetime.now().weekday() + 1
                hour = datetime.datetime.now().time()
                inSchedule = self.searchInSchedule(weekday, hour, self.config['powers'])
                print('runing on: ', inSchedule)

            if inSchedule:
                t1 = cv2.getTickCount()
                grabbed, realframe = self.videostream.read()
                # Compruebo que me llega imagen
                if not grabbed or realframe is None:
                    if cv2.waitKey(20) == ord('q'):
                        break
                    time.sleep(0.01)
                    continue
                try:
                    frame = realframe[self.Y_start : self.Y_end, self.X_start : self.X_end]
                except:
                    if cv2.waitKey(20) == ord('q'):
                        break
                    break
                # Puedo variar el frame rate con detect_interval
                if cnt % detect_interval == 0:
                    if not grabbed or realframe is None:
                        if cv2.waitKey(20) == ord('q'):
                            break
                        time.sleep(0.01)
                        continue
                    try:
                        (self.imH, self.imW) = realframe.shape[:2]
                    except:
                        if cv2.waitKey(20) == ord('q'):
                            break
                    #st = time.time()
                    # Detecto las caras + género + edad + embedding
                    # [:,:,::-1] esto es debido a que opencv trabaja con BGR
                    face_list = []
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_list = self.detector.get(frame_rgb)
                        cnt += 1
                    except:
                        if cv2.waitKey(20) == ord('q'):
                            break

                    # if we have detections
                    if len(face_list) > 0:
                        for f in face_list:
                            # compruebo que las caras detectadas cumplan con la calidad
                            # Ver en face_quality otros posibles params de calidad que se pdorían aplicar
                            # checkeo el threshold del modelo
                            if f.det_score < self.configs["face_threshold"]:
                                break
                            # checkeo el embedding norm
                            if f.embedding_norm < self.configs["embedding_threshold"]:
                                break
                            box = f.bbox.astype(np.intc)
                            # checkeo el tamaño de la cara
                            if box[2]-box[0] < self.configs["face_size"] or  box[3]-box[1] < self.configs["face_size"]:
                                break
                            face_image = frame[box[1]:box[3], box[0]:box[2]]
                            ####################################
                            # distance measurement comparo la nueva cara con todas las demás
                            ####################################
                            # First detection
                            if not all_embeddings:
                                distCosine = False
                            else:
                                # comparo distancia entre el embed de la cara detectada y todos los embeds de las caras ya detectadas
                                # ID es la posición
                                distCosine, ID = findCosineskLearn(all_embeddings.copy(),f.normed_embedding,self.configs["cosine_threshold"])

                            face = None
                            # si la distancia no supera el threshold se trata de nueva persona
                            if distCosine is False:
                                new_person += 1
                                face = face_store.add(self.Ids,box,f.embedding_norm, f.sex, f.age)
                                face_image = frame[box[1]:box[3], box[0]:box[2]]
                                self.sendFaceIn(face_store.get(self.Ids),face_image)
                                all_embeddings.append((str(self.Ids),f.normed_embedding))
                                self.Ids += 1
                                ## Save faces images
                            else:
                                # Update the current ID face with the new detection details
                                face = face_store.add(str(ID),box,f.embedding_norm, f.sex, f.age)
                                all_embeddings.append((str(ID),f.normed_embedding))
                            ####################################
                            # painting results
                            ####################################
                            color = (0, 0, 255)
                            pos = [box[0] + self.X_start, box[1] + self.Y_start, box[2] + self.X_start, box[3] + self.Y_start]
                            cv2.rectangle(realframe, (pos[0], pos[1]), (pos[2], pos[3]), color, 2)
                            text = str(ID)
                            cv2.putText(realframe,text, (pos[0]-1, pos[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
                            if f.age < 10:
                                age_type = 'less than 10'
                            elif f.age >= 10 and f.age < 20 :
                                age_type = 'between 10 and 20'
                            elif f.age >= 20 and f.age < 30 :
                                age_type = 'between 20 and 30'
                            elif f.age >= 30 and f.age < 40 :
                                age_type = 'between 30 and 40'
                            elif f.age >= 40 and f.age < 50 :
                                age_type = 'between 40 and 50'
                            elif f.age >= 50 and f.age < 60 :
                                age_type = 'between 50 and 60'
                            elif f.age >= 60 and f.age < 70 :
                                age_type = 'between 60 and 70'
                            else:
                                age_type = 'older than 70'

                            sex_type = 'Man' if f.sex == 'M' else 'Woman'                            
                            # cv2.putText(realframe,'%s, %s'%(sex_type,age_type), (pos[2]-1, pos[3]-4),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,255,0),1)
                            
                            tags = face_store.getTagsToSend()
                            cv2.putText(realframe,tags[0]['name'], (pos[2]-1, pos[3]-4),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,255,0),1)

                t2 = cv2.getTickCount()
                time1 = (t2 - t1) / freq
                total_time += time1
                frame_rate_calc = cnt / total_time

                text_color = (0, 255, 50)
                if self.data_detector['show_time'] == 'True':
                    hora = time.ctime() # 'Mon Oct 18 13:35:29 2010'
                    hora = time.strftime('%X')

                    width = self.imW - 10
                    text = 'People:{0:}, time: {1:}'.format(new_person,hora)
                    fontSize = self.get_optimal_font_scale(text, width)
                    cv2.putText(realframe, text, (5, int(fontSize * 30)), cv2.FONT_HERSHEY_SIMPLEX, fontSize, text_color, 2, cv2.LINE_AA)
                else:
                    width = self.imW - 10
                    text = 'People:{0:}, time: {1:}'.format(new_person,hora)
                    fontSize = self.get_optimal_font_scale(text, width)
                    cv2.putText(realframe, 'People:{0:}'.format(new_person), (5, int(fontSize * 30)), cv2.FONT_HERSHEY_SIMPLEX, fontSize, text_color, 2, cv2.LINE_AA)
                title = 'Face detector | ' + email
                if self.data_detector['show_video'] == 'True':
                    color = (0, 125, 255)
                    cv2.rectangle(realframe, (self.X_start, self.Y_start), (self.X_end, self.Y_end), color, 2)
                    cv2.imshow(title, realframe)
                if self.start_stream_time > 0:
                    if time.time() - self.start_stream_time < STREAM_TIME:
                        try:
                            if self.checkStream(): 
                                self.p.stdin.write(frame.tostring())
                            else:
                                self.stopStream()
                        except Exception as e:
                            self.stopStream()

                    else:   # stop stream
                        print('Steaming Stop Automatically')
                        self.stopStream()

                        send_data = {
                            'from': self.sio.sid,
                            'to': 'ids.ladorian.camera.' + str(self.config['idcamera']),
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

    def searchInSchedule(self,weekday, hour, powers):
        for power in powers:
            if power['weekday'] == weekday:
                start = datetime.datetime.strptime(power['time_on'], '%H:%M:%S').time()
                end = datetime.datetime.strptime(power['time_off'], '%H:%M:%S').time()
                if start <= hour and end >= hour:
                    return True
        return False

    def get_optimal_font_scale(self, text, width):
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return min(scale/10, 1)
        return 1

    def setThreadInfo(self, sio, config):
        self.sio = sio
        self.config = config
        self.age_gender_proc.setThreadInfo(sio, config)

    def stop(self):
        self.thread_running = False
        self.age_gender_proc.stop()

    def sendFaceIn(self, face, faceImage):
        print("Face In: ID = ", str(face.id))
        #retval, buffer = cv2.imencode('.jpg', faceImage)
        #jpg_as_text = base64.b64encode(buffer)

        send_data = {
            'from': self.sio.sid,
            'to': 'ids.ladorian.camera.' + str(self.config['idcamera']),
            'name': 'face:in',
            'data': {
                'id': str(face.id),
                'gender': face.gender,
                'age': face.age,
            }
            #'image': jpg_as_text,
        }

        if self.sio is not None:
            self.sio.emit('send:message', send_data)

    def startStream(self, socketid):

        if self.checkStream() == False:
            self.stopStream()

        if self.start_stream_time > 0:
            return self.rtmp_key
        
        dt = datetime.datetime.now()
        utc_time = dt.replace(tzinfo=timezone.utc)
        ts = int(utc_time.timestamp() * 1000)
        self.rtmp_key = socketid + str(ts)
        
        rtmp_url = 'rtmp://streaming.ladorianids.es:1935/live/' + self.rtmp_key
        command = ['ffmpeg',
                        '-v', 'error',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(self.imW, self.imH),
                        '-r', str(self.videostream.fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast',
                        '-f', 'flv',
                        rtmp_url]

        # start stream
        print('Steaming Start = ' + rtmp_url)
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.start_stream_time = time.time()
        self.age_gender_proc.setStartStreamTime(self.start_stream_time)
        return self.rtmp_key

    def checkStream(self):
        try:
            if self.p:
                if self.p.stderr:
                    raise Exception('Streaming error')
                else:
                    return True
        except Exception as e:
            print(e)

        return False

    def stopStream(self):
        try:
            if self.p:
                self.p.stdin.close()
                self.p.kill()
        except Exception as e:
            print(e)

        self.start_stream_time = 0
        self.age_gender_proc.setStartStreamTime(self.start_stream_time)
