from datetime import timezone
import datetime
import requests
import hashlib
import socketio
from device import Device
import json
import time
import sys
from camera_control import CameraThread
from threading import Timer

def getserial():
  # Extract serial from cpuinfo file
  cpuserial = "0000000000000000"
  try:
    f = open('/proc/cpuinfo','r')
    for line in f:
      if line[0:6]=='Serial':
        cpuserial = line[10:26]
    f.close()
  except:
    cpuserial = "ERROR000000000"

  return cpuserial

print("1. Launch App")

config = {}

def readConfig():
    try:
        with open('config.ini') as json_file:
            data = json.load(json_file)
        with open('config_detector.ini') as json_file:
            data_detector = json.load(json_file)
    except Exception as e:
        print(e)
        data= {}
        with open('config_detector.ini') as json_file:
            data_detector = json.load(json_file)
        data['email'] = ''
        print('Falta archivo de configuración')

    return data,data_detector


def saveConfig(config_data):
    try:
        with open('config.ini', 'w') as outfile:
            json.dump(config_data, outfile)
    except:
        print('Write Exception')


config,data_detector = readConfig()

sio = socketio.Client(False)
#le añado pasarle la config en el init para uso local
camera_thread = CameraThread(config,data_detector)

def generateKeyHash():
    dt = datetime.datetime.now()
    utc_time = dt.replace(tzinfo=timezone.utc)
    ts = int(utc_time.timestamp() * 1000)
    key = str(ts) + "ALRIDKJCS1SYADSKJDFS"
    h = hashlib.sha256(key.encode('utf-8'))
    keyhash = h.hexdigest()

    return ts, keyhash

def sendError(message):
    send_data = {
        'from': sio.sid,
        'to': 'ids.ladorian.camera',
        'name': 'log:error',
        'message': message
    }
    sio.emit('send:message', send_data)

def reboot():
    try:
        if camera_thread.is_alive():
            camera_thread.stopStream()
            camera_thread.stop()
    except:
        print("Close camara thread ERROR")

    #Device.reboot()

def reconnect():
    try:
        print('2. Connection Start')
        sio.connect('https://pusher.ladorianids.com')
    except:
        print('Socket exception')
        #sio = socketio.Client(False)
        t = Timer(10.0, reconnect)
        t.start()

def startProc():
    if config['email'] is None or config['email'] == '':
        arg_email = sys.argv[1]
        if(arg_email is None or arg_email == ''):
            print('4. Log Error is sent')
            sendError('email is empty')
        else:
            data = {'email': arg_email}
            auth(data)
    else:
        auth(config)

@sio.event
def connect():
    print('3. Socket is connected = ', sio.sid)
    if camera_thread.is_alive():
        camera_thread.stopStream()

    startProc()

@sio.event
def connect_error():
    print("Socket connection failed!")

    t = Timer(5.0, reconnect)
    t.start()

@sio.event
def disconnect():
    print("I'm disconnected!")

    t = Timer(5.0, reconnect)
    t.start()


@sio.on('message')
def on_message(data):
    if data.get('name') and data['name'] == 'set:email':
        config['email'] = data['data']
        auth(config)

    if data.get('name') and data['name'] == 'app:restart':
        # send restart log info
        send_data = {
            'from': sio.sid,
            'to': 'ids.ladorian.camera',
            'name': 'log:info',
            'message': 'device is restarted'
        }
        sio.emit('send:message', send_data)

        startProc()

    if data.get('name') and data['name'] == 'get:status':
        channel_config = camera_thread.config
        data_obj = {
            'status': True,
            'email': channel_config['email'],
            'camera_id': str(channel_config['idcamera']),
            'site_id': str(channel_config['site']['idsite']),
            'site_name': channel_config['site']['name'],
            'customer_id': str(channel_config['customer']['idcustomer']),
            'customer_name': channel_config['customer']['name'],
            'powers': channel_config['powers'],
            'holidays': [],
        }
        if camera_thread.is_alive() and channel_config is not None:
            data_obj['status'] = True
        else:
            data_obj['status'] = False

        send_data = {
            'from': sio.sid,
            'to': data['from'],
            'name': 'status',
            'data': data_obj
        }

        sio.emit('send:message', send_data)

    if data.get('event') and data['event'] == 'app.status':
        channel_config = camera_thread.config
        data_obj = {
            'status': True,
            'email': channel_config['email'],
            'camera_id': str(channel_config['idcamera']),
            'site_id': str(channel_config['site']['idsite']),
            'site_name': channel_config['site']['name'],
            'customer_id': str(channel_config['customer']['idcustomer']),
            'customer_name': channel_config['customer']['name'],
            'powers': channel_config['powers'],
            'holidays': [],
        }
        if camera_thread.is_alive() and channel_config is not None:
            data_obj['status'] = True
        else:
            data_obj['status'] = False

        send_data = {
            'from': sio.sid,
            'to': data['from'],
            'event': 'app.status',
            'data': data_obj
        }

        sio.emit('send:message', send_data)


    if data.get('name') and data['name'] == 'streaming:open':
        # URL donde se manda el streaming de vídeo
        rtmp_key = camera_thread.startStream(sio.sid)
        flv_url = 'https://streaming.ladorianids.es/flv?port=1935&app=live&stream=' + rtmp_key

        exists = False
        # while exists == False:
        #     print("no existe")
        #     exists = uri_exists_stream(flv_url)

        print("Ya existe")
        send_data = {
            'from': sio.sid,
            'to': data['from'],
            'name': 'streaming:opened',
            'data': flv_url
        }
        sio.emit('send:message', send_data)

def uri_exists_stream(uri: str) -> bool:
    try:
        with requests.get(uri, stream=True) as response:
            try:
                response.raise_for_status()
                return True
            except requests.exceptions.HTTPError:
                return False
    except requests.exceptions.ConnectionError:
        return False

def auth(data):
    ts, keyhash = generateKeyHash()
    url = 'https://ids.ladorianids.es/protected/camera/auth?email=' + data['email'] + '&timestamp=' + str(ts) + '&keyhash=' + keyhash
    channel_config = None
    while True:
        try:
            response = requests.get(url)
            ret = response.json()
            if ret['success'] == True:
                print('Auth success')
                channel_config = ret['data']
                break
            else:
                errMsg = ret['data']
                print('Auth error', errMsg)
                sendError(errMsg)
        except:
            print('Auth exception')
            time.sleep(5)

    # save config
    saveConfig(channel_config)

    # join channel
    sio.emit('join:room', 'camera.id.' + str(channel_config['idcamera']))
    sio.emit('join:room', 'camera.customer.' + str(channel_config['customer']['idcustomer']))
    sio.emit('join:room', 'camera.site.' + str(channel_config['site']['idsite']))
    sio.emit('join:room', 'camera.email.' + channel_config['email'])
    
    camera_thread.setThreadInfo(sio, channel_config)
    if not camera_thread.is_alive():
        camera_thread.start()

    # if activated
    if camera_thread.is_alive():
        send_data = {
            'from': sio.sid,
            'to': 'ids.ladorian.camera',
            'name': 'log:info',
            'message': 'device is activated'
        }
        sio.emit('send:message', send_data)
reconnect()
print("End Program")
