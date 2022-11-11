#!/usr/bin/env python3

import cv2
from enum import Enum

STATE = Enum('STATE', ['Continue', 'Stop'])
PATH = 'bricks/Bricks_wide-04.mp4'
WINDOW_NAME = 'Frame'
KEY_Q = 113

FRAME_SIZE = (1920//3, 1080//3)
FRAME_DELAY = 60


def process_video(path):
  vidcap = cv2.VideoCapture(path)

  while True:
    success, image = vidcap.read()
    if not success: break

    frame = cv2.resize(image.copy(), FRAME_SIZE)

    cv2.imshow(WINDOW_NAME, frame)
    cv2.moveWindow(WINDOW_NAME, 1920, 0)
    key = cv2.waitKey(FRAME_DELAY)

    if key is KEY_Q: return STATE.Stop

  return STATE.Continue


while True:
  state = process_video(PATH)
  if state is state.Stop: break

cv2.destroyAllWindows()
