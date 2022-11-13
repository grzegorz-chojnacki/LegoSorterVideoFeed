#!/usr/bin/env python3

import cv2
import grpc
from enum import Enum
from generated import LegoSorter_pb2_grpc as LegoSorter
from generated.Messages_pb2 import ImageRequest, ListOfBoundingBoxes


STATE = Enum('STATE', ['Continue', 'Stop'])
PATH = 'bricks/Bricks_wide-04.mp4'
WINDOW_NAME = 'Frame'
KEY_Q = 113

FRAME_SIZE = (640, 640)
FRAME_MARGIN = 30
FRAME_DELAY = 120


class RemoteLegoBrickSorter(object):
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f'{self.host}:{self.server_port}')
        self.stub = LegoSorter.LegoSorterStub(self.channel)

    def prepareImageMessage(self, image):
        return ImageRequest(image=cv2.imencode('.jpg', image)[1].tobytes())

    def processNextImage(self, image):
        return self.stub.processNextImage.future(self.prepareImageMessage(image))


def crop(image):
    height, width, _ = image.shape
    h_crop = FRAME_MARGIN
    w_crop = ((width - height) // 2) + FRAME_MARGIN
    frame = image[h_crop:-h_crop, w_crop:-w_crop]
    return cv2.resize(frame.copy(), FRAME_SIZE)


def process_video(path):
    global frame_sent
    global sorter

    vidcap = cv2.VideoCapture(path)

    while True:
        success, image = vidcap.read()
        if not success:
            break

        frame = crop(image)

        # Send request
        sorter.processNextImage(frame)
        frame_sent += 1
        print(f'Sent frame #{frame_sent}')

        # Show frame in the meantime
        cv2.imshow(WINDOW_NAME, frame)
        cv2.moveWindow(WINDOW_NAME, 1920, 0)
        key = cv2.waitKey(FRAME_DELAY)

        if key is KEY_Q:
            return STATE.Stop

    return STATE.Continue


if __name__ == '__main__':
    sorter = RemoteLegoBrickSorter()
    frame_sent = 0

    while True:
        state = process_video(PATH)
        if state is STATE.Stop:
            break

    cv2.destroyAllWindows()
