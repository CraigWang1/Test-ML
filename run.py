#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time
import utils
from tensorflow.python.platform import gfile

if __name__ == "__main__":
  utils.inference_vid(
    model_path="model.pb",
    phi=0,
    input_vid="video.mp4",
    output_vid="result.mp4",
    score_threshold=0.5,
    classes=["gate"],
    fps=25
  )