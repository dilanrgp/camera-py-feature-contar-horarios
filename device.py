import os

class Device:

    @staticmethod
    def reboot():
        try:
            os.system('reboot now')
        except:
            print('Error reboot')
