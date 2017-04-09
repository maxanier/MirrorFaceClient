from time import sleep
from threading import Thread
import server
import constants
import config
import detector

if __name__ == '__main__':
    print("Starting MirrorFace Client")
    detector.init()

    server_thread= Thread(target=server.run_http_api)
    server_thread.start()

    while True:
        if config.state == constants.STATUS_RUNNING:
            print("Detected: {0}".format(detector.check()))

        sleep(1)

