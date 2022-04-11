#!/usr/bin/python

import random
import math
import argparse
import os
import cv2
import dlib

from cv2 import dnn_superres

#import numpy as np

# scale_factor = 1.1
# min_neighbors = 6
#
# border = 0.5
#
# filter = "haze"
#
# line_num = 150
# line_color = (0, 0, 0)

upscale_size = 0


def rand_point(w, h):
    theta = random.random() * math.tau
    scale = random.triangular(0, 1, 0.5)
    x = math.sin(theta) * int(scale * w / 2) + w / 2
    y = math.cos(theta) * int(scale * h / 2) + h / 2
    return (int(x), int(y))


def lerp_color(c1, c2, s):
    s = max(0, min(s, 1))
    p1 = c1[0] * s + c2[0] * (1 - s)
    p2 = c1[1] * s + c2[1] * (1 - s)
    p3 = c1[2] * s + c2[2] * (1 - s)
    return (p1, p2, p3)


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def process_face(input, args):
    w = input.shape[0]
    h = input.shape[1]

    if args.filter == "haze":
        for x in range(w):
            for y in range(h):
                r = random.randint(0, 255)
                color = (r, r, r)
                shift = (1 - (dist(x / w, y / h, 0.5, 0.5) * 2)) * 1.5
                input[x][y] = lerp_color(color, input[x][y], shift)

    elif args.filter == "scratch":
        for i in range(args.line_num):
            p1 = rand_point(w, h)
            p2 = rand_point(w, h)
            thickness = random.randint(2, 4)
            input = cv2.line(input, p1, p2, args.line_color,
                             thickness, cv2.LINE_AA)

    return input


def process_image(input_name, output_name, args):
    if args.verbose >= 1:
        print("Processing", input_name)

    image = cv2.imread(input_name)

    if args.upscale:
        sr = dnn_superres.DnnSuperResImpl_create()
        path = "Upscale/FSRCNN_x2.pb"
        sr.readModel(path)
        sr.setModel("fsrcnn", 2)

        image_scaled = sr.upsample(image)
        cv2.imwrite("tmp.png", image_scaled)
        image = cv2.imread("tmp.png")

        if args.verbose >= 2:
            print("Upscaled")

    if args.mode == "haar":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        model = cv2.CascadeClassifier(args.model)
        faces = model.detectMultiScale(
            gray, args.scale_factor, args.min_neighbors)

    elif args.mode == "cnn":
        detector = dlib.cnn_face_detection_model_v1(args.model)
        img = dlib.load_rgb_image("tmp.png" if args.upscale else input_name)
        faces = detector(img, upscale_size)

        faces = list(map(lambda f: (f.rect.left(), f.rect.top(),
                     f.rect.width(), f.rect.height()), faces))

    else:
        raise(ValueError)

    if args.verbose >= 2:
        print("Found", len(faces), "faces")

    for (x, y, w, h) in faces:
        a = max(int(y - h * args.border), 0)
        b = min(int(y + h + h * args.border), image.shape[0])
        c = max(int(x - w * args.border), 0)
        d = min(int(x + w + w * args.border), image.shape[1])

        ROI = image[a:b, c:d]

        processed = process_face(ROI, args)

        image[a:b, c:d] = processed

    # Save the result image
    cv2.imwrite(output_name, image)

    if args.upscale:
        os.remove("tmp.png")


def get_args():
    parser = argparse.ArgumentParser(
        description="Distort all faces in a photo")

    parser.add_argument(
        "input", type=str, help="Image or directory of images to be processed")
    parser.add_argument("dest", nargs="?", type=str,
                        default="processed.png", help="Destination file")
    parser.add_argument("-d", dest="dir", action="store_true",
                        help="Set this flag if passing a directory as input")
    parser.add_argument("-m", "--model", nargs="?",
                        type=str, default="Models/face.xml",
                        help="Face detection model to be used. Can be either and xml file for OpenCV's Haar Cascade detection or a dat file for dlib's Convolutional Neural Network")
    parser.add_argument("-s", "--scale-factor", type=float, default=1.1,
                        help="Scaling factor for Haar Cascade detection")
    parser.add_argument("-n", "--min-neighbors", type=int, default=6,
                        help="Minimum number of neighbors for a positive Haar Cascade detection")
    parser.add_argument("-b", "--border", type=float, default=0.5,
                        help="Percentage border to add the the size of each detected rectangle")
    parser.add_argument("-f", "--filter", type=str, default="haze",
                        help="Filter to aplly to all detected faces. Options are Haze and Scratch")
    parser.add_argument("-l", "--line-num", type=int, default=150,
                        help="Number of lines to add for the Scratch filter")
    parser.add_argument("-c", "--line-color", type=int,
                        nargs=3, default=(0, 0, 0), help="Color of lines to add for the Scratch filter")
    parser.add_argument("-v", "--verbose", action="count", default=0)

    parser.add_argument("-u", dest="upscale", action="store_true",
                        help="Set this flag if the image should be upscaled")

    args = parser.parse_args()

    args.filter = args.filter.lower()

    if os.path.splitext(args.model)[1] == ".xml":
        args.mode = "haar"
    elif os.path.splitext(args.model)[1] == ".dat":
        args.mode = "cnn"

    return args


def main():
    args = get_args()

    if args.verbose >= 1 and args.mode == "cnn":
        if dlib.DLIB_USE_CUDA:
            print("Using CUDA")
        else:
            print("Not using CUDA")

    if args.dir:
        i = 1
        for file in os.scandir(args.input):
            if not file.is_file():
                continue

            input_file = file.path

            output_parts = os.path.splitext(args.dest)
            output_file = output_parts[0] + str(i) + output_parts[1]

            process_image(input_file, output_file, args)

            i += 1
    else:
        input_file = args.input
        output_file = args.dest

        process_image(input_file, output_file, args)


if __name__ == "__main__":
    main()
