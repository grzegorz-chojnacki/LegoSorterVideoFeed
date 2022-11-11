#!/usr/bin/env python3

import cv2
import grpc
from enum import Enum
from generated import LegoAnalysis_pb2_grpc as LegoAnalysis
from generated.Messages_pb2 import ImageRequest, ListOfBoundingBoxes


STATE = Enum('STATE', ['Continue', 'Stop'])
PATH = 'bricks/Bricks_wide-04.mp4'
WINDOW_NAME = 'Frame'
KEY_Q = 113

FRAME_SIZE = (640, 640)
FRAME_MARGIN = 30
FRAME_DELAY = 120


class RemoteLegoBrickAnalyzer(object):
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f'{self.host}:{self.server_port}')
        self.stub = LegoAnalysis.LegoAnalysisStub(self.channel)

    def prepareImageMessage(self, image):
        return ImageRequest(image=cv2.imencode('.jpg', image)[1].tobytes())

    def detectBricks(self, image):
        return self.stub.DetectBricks.future(self.prepareImageMessage(image))

    def detectAndClassifyBricks(self, image):
        return self.stub.DetectAndClassifyBricks.future(self.prepareImageMessage(image))


def crop(image):
    height, width, _ = image.shape
    h_crop = FRAME_MARGIN
    w_crop = ((width - height) // 2) + FRAME_MARGIN
    frame = image[h_crop:-h_crop, w_crop:-w_crop]
    return cv2.resize(frame.copy(), FRAME_SIZE)


def process_video(path):
    global frame_sent
    global analyzer

    vidcap = cv2.VideoCapture(path)

    while True:
        success, image = vidcap.read()
        if not success:
            break

        frame = crop(image)

        # Send request
        analyzer.detectAndClassifyBricks(frame)
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
    analyzer = RemoteLegoBrickAnalyzer()
    frame_sent = 0

    while True:
        state = process_video(PATH)
        if state is state.Stop:
            break

    cv2.destroyAllWindows()
