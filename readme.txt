# Install/reinstall Coral's libedgetpu.
# https://github.com/leigh-johnson/rpi-deep-pantilt/issues/37
$ echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

$ curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install libedgetpu1-std

sudo apt-get install libcblas-dev
sudo apt-get install libhdf5-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libjasper-dev 
sudo apt-get install libqtgui4 
sudo apt-get install libqt4-test
sudo apt-get install libedgetpu1-std

git clone https://github.com/OscarHP/FTR_raspberry.git
sudo pip3 install virtualenv
virtualenv cv -p python3
cd FTR_raspberry
source cv/bin/activate

pip3 install -r requirements.txt
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

# enable camera
# https://projects.raspberrypi.org/en/projects/getting-started-with-picamera/2

python main.py


