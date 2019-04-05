'''
Adapted from the MonoHand3D codebase for the MonocularRGB_2D_Handjoints_MVA19 project (github release)

This is a simple sample script for running the pretrained network for fast 2D joint estimation
presented in "Accurate Hand Keypoint Localization on Mobile Devices" (MVA19)

Only heatmaps are recovered (raw network output). If you need to extract keypoints check the code in this project:
https://github.com/FORTH-ModelBasedTracker/MonocularRGB_3D_Handpose_WACV18


@author: Paschalis Panteleris (padeler@ics.forth.gr)
'''

from __future__ import print_function
from common.mva19 import Estimator, preprocess
import numpy as np
import cv2
import time


if __name__ == "__main__":

    model_file = "./models/mobnet4f_cmu_adadelta_t1_model.pb"
    input_layer = "input_1"
    output_layer = "k2tfout_0"

    stride = 4
    boxsize = 224

    estimator = Estimator(model_file, input_layer, output_layer)

    # start webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.4)

    paused = True
    delay = {False: 1, True: 0}


    k = 0
    while k != ord('q'):
        ret, frame = cap.read()
        if not ret:
            raise Exception("VideoCapture.read() returned False")

        crop_res = cv2.resize(frame, (boxsize, boxsize))
        img, pad = preprocess(crop_res, boxsize, stride)

        tic = time.time()
        hm = estimator.predict(img)
        dt = time.time() - tic
        print("TTP %.5f, FPS %f" % (dt, 1.0/dt), "HM.shape ", hm.shape)

        hm = cv2.resize(hm, (0, 0), fx=stride, fy=stride)
        bg = cv2.normalize(hm[:, :, -1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        viz = cv2.normalize(np.sum(hm[:, :, :-1], axis=2), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("Background", bg)
        cv2.imshow("All joint heatmaps", viz)
        cv2.imshow("Input frame", frame)

        k = cv2.waitKey(delay[paused])

        if k & 0xFF == ord('p'):
            paused = not paused
