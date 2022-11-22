#!/usr/bin/env python3

import cv2
import grpc
import time
import threading
from enum import Enum
from generated import LegoSorter_pb2_grpc as LegoSorter
from generated.Messages_pb2 import ImageRequest, ListOfBoundingBoxes


STATE = Enum('STATE', ['Continue', 'Stop'])
PATH = 'bricks/Bricks_wide-04.mp4'
WINDOW_NAME = 'Frame'
KEY_Q = 113

FRAME_SIZE = (640, 640)
FRAME_MARGIN = 30
FRAME_DELAY = 300


class RemoteLegoBrickSorter(object):
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051
        self.channel = grpc.insecure_channel(f'{self.host}:{self.server_port}')
        self.stub = LegoSorter.LegoSorterStub(self.channel)

    def prepareImageMessage(self, image):
        return ImageRequest(image=cv2.imencode('.jpg', image)[1].tobytes())

    def processNextImageSync(self, image, frame_n):
        start = time.time()
        result = self.stub.processNextImage(self.prepareImageMessage(image))
        end = time.time()
        print(f'Processing frame #{frame_n} took {(end - start) * 1000} ms')
        return result

    async def processNextImageAsync(self, image, frame_n):
        start = time.time()
        result = await self.stub.processNextImage.future(self.prepareImageMessage(image))
        end = time.time()
        print(f'Processing frame #{frame_n} took {(end - start) * 1000} ms')
        return result


def crop(image):
    height, width, _ = image.shape
    h_crop = FRAME_MARGIN
    w_crop = ((width - height) // 2) + FRAME_MARGIN
    frame = image[h_crop:-h_crop, w_crop:-w_crop]
    return cv2.resize(frame.copy(), FRAME_SIZE)

def process_video_sync():
    sorter = RemoteLegoBrickSorter()
    video = cv2.VideoCapture(PATH)
    frame_n = 0

    while True:
        frame_n += 1
        success, image = video.read()
        if not success:
            break

        if frame_n < 30:
            continue

        frame = crop(image)

        sorter.processNextImageSync(frame, frame_n)

        # Show frame in the meantime
        cv2.imshow(WINDOW_NAME, frame)
        cv2.moveWindow(WINDOW_NAME, 0, 0)
        key = cv2.waitKey(FRAME_DELAY)

        if key is KEY_Q:
            break

    cv2.destroyAllWindows()


def process_video_async():
    sorter = RemoteLegoBrickSorter()
    video = cv2.VideoCapture(PATH)
    frame_n = 0
    threads = []

    while True:
        frame_n += 1
        success, image = video.read()

        if not success:
            break

        if frame_n < 30:
            continue

        t = threading.Thread(target=lambda: sorter.processNextImageSync(crop(image), frame_n))
        threads.append(t)
        t.start()
        alive = len(list(filter(lambda t: t.is_alive(), threads)))
        dead = len(list(filter(lambda t: not t.is_alive(), threads)))
        print(f'Sent frame #{frame_n:03} | Threads alive/dead: {alive}/{dead}')
        time.sleep(FRAME_DELAY / 1000)


if __name__ == '__main__':
    # process_video_sync()
    process_video_async()
