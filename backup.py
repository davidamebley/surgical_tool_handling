# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import time
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle
import math
import cv2
import torch
import torch.backends.cudnn as cudnn
from scipy.spatial import distance
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from timeit import default_timer
from tasks import path_tracking as pk
from gui import app

# Custom variables
no_of_task_repeats = 2  # How many times to repeat the task in a session
current_task_iteration = 0

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# # Check if path tracking started
# is_start_path_tracking = False
#
# # Check if timer started
# is_timer_started = False
#
# # Check if task complete
# is_task_complete = False
#
# # Check if tracker has started moving
# is_tracker_start = False
#
# # Other Custom Variables
# x, y, im0 = 0, 0, 0
# # Custom variables
# path_length = 0
# tracker_path_threshold = 10  # 10
# tracker_tooltip_threshold = 30  # 10
# tracker_end_point_threshold = 300  # 3
# poi = []
# tool_tip = []
# tracker = []
# tooltip_dist = 0
# last_tracker_position = []
# points_traversed = []
# distance_travelled = 0
# elapsed_time = 0  # time taken to complete task
# start_time = 0  # time task started
# overall_deviation = 0  # Deviation from ideal path (in distance)
#
# # pts = []
# pts = [(424, 305), (418, 329), (408, 346), (405, 326), (405, 300), (405, 274), (408, 246), (408, 222), (407, 196),
#        (415, 163), (404, 136), (401, 109), (386, 86), (379, 64), (365, 87), (358, 108), (347, 131), (329, 156),
#        (318, 186), (307, 208), (295, 231), (287, 256), (281, 284), (276, 311), (271, 340), (269, 315), (246, 299),
#        (233, 272), (223, 250), (207, 228), (199, 202), (194, 169), (183, 142), (170, 115), (164, 92), (161, 63)]
#
# img_poly = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
#
#
# def draw_roi(event, x, y, flags, param):
#     img2 = img_poly.copy()
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         pts.append((x, y))
#         # print(x,y)
#     if event == cv2.EVENT_RBUTTONDOWN:
#         pts.pop()
#
#     if event == cv2.EVENT_MBUTTONDOWN:
#         mask = np.zeros(img_poly.shape, np.uint8)
#         points = np.array(pts, np.int32)
#         points = points.reshape((-1, 1, 2))
#
#         mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
#         mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
#         mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))
#
#         show_image = cv2.addWeighted(src1=img_poly, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
#
#         cv2.imshow("mask", mask2)
#         cv2.imshow("show_img", show_image)
#
#         ROI = cv2.bitwise_and(mask2, img_poly)
#         cv2.imshow("ROI", ROI)
#         cv2.waitKey(0)
#
#     if len(pts) > 0:
#         cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
#
#     if len(pts) > 1:
#
#         for i in range(len(pts) - 1):
#             cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
#             cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
#
#     cv2.imshow('image', img2)
#     return pts, cv2.imshow('image', img2)
#
#
# def draw_tracker(im, point_x, point_y, original_width=1920, original_height=1080):
#     # Draw Tracker circle
#     cv2.circle(im, (point_x * original_width // 640, point_y * original_height // 480), 7, (0, 0, 255), 7)
#     # pts[0][0] * w // 640, pts[0][1] * h // 480
#     update_variables(index=1, t_value=(point_x, point_y))
#
#
# def draw_tool_tip(im, point_x, point_y, original_width=1920, original_height=1080):
#     # Draw Tracker circle
#     cv2.circle(im, (point_x * original_width // 640, point_y * original_height // 480), 3, [0, 255, 255], 5)
#     # int((x / w) * 640), int((y / h) * 480)
#     update_variables(index=0, t_value=(point_x // original_width * 640, point_y // original_height * 480))
#
#
# # Update My Tracking Variables
# def update_variables(index, t_value):
#     global tool_tip, tracker
#     if index == 0:
#         tool_tip = t_value
#     elif index == 1:
#         tracker = t_value
#     return t_value
#
#
# # Function to determine the on_line status of tracker
# def is_tracker_on_line(tracker):
#     a, temp = 0, 0
#     for n in range(len(pts)):
#         if n < len(pts) - 1:
#             coord_x1, coord_y1 = pts[n]  # Current point
#             coord_x2, coord_y2 = pts[n + 1]  # Next point
#             dist = abs((coord_y2 - coord_y1) * tracker[0] - (coord_x2 - coord_x1) * tracker[
#                 1] + coord_x2 * coord_y1 - coord_y2 * coord_x1) / math.sqrt(
#                 (coord_y2 - coord_y1) ** 2 + (coord_x2 - coord_x1) ** 2)
#             if n == 0:
#                 a = dist
#             else:
#                 temp = dist
#                 if temp < a:
#                     a = temp
#             print("current distance = ", dist)
#     return a
#
#
# # Function to check the shortest distance between tracker and drawn path
# def check_tracker_distance(m_tracker):
#     try:
#         trk_x = m_tracker[0]
#         trk_y = m_tracker[1]
#         short, temp = 0, 0
#         for n in range(len(pts)):
#             if n < len(pts) - 1:
#                 x1, y1 = pts[n]
#                 x2, y2 = pts[n + 1]
#                 a = trk_x - x1
#                 b = trk_y - y1
#                 c = x2 - x1
#                 d = y2 - y1
#
#                 dot = a * c + b * d
#                 len_sq = c * c + d * d
#                 param = -1
#
#                 if len_sq != 0:  # in case of 0 length line
#                     param = dot / len_sq
#
#                 if param < 0:
#                     xx = x1
#                     yy = y1
#                 elif param > 1:
#                     xx = x2
#                     yy = y2
#                 else:
#                     xx = x1 + param * c
#                     yy = y1 + param * d
#
#                 dx = trk_x - xx
#                 dy = trk_y - yy
#                 dist = math.sqrt(dx * dx + dy * dy)
#                 # Store the shortest distance
#                 if n < 1:
#                     short = dist
#                 else:
#                     temp = dist
#                     if temp < short:
#                         short = temp  # shortest distance
#                 print("distance at ", pts[n], " and ", pts[n + 1], "===:: ", dist, " .shortest dist is ", short)
#         return short
#     except:
#         print("ERROR in checking tracker distance to path")
#
#
# # Function to generate total length of drawn path
# def get_path_length():
#     length = 0
#     try:
#         for n in range(len(pts)):
#             if n < len(pts) - 1:
#                 length += distance.euclidean(pts[n], pts[n + 1])
#         return length
#     except:
#         print("invalid path length")
#
#
# # Function to start timing performance
# def start_timer():
#     global is_timer_started, start_time
#     if not is_timer_started:
#         # start_time = time.time()
#         start_time = default_timer()  # default_timer is independent of unfortunate system time's changes during usage
#         is_timer_started = True
#         return start_time
#
#
# # Function to stop performance timer
# def stop_timer():
#     global is_timer_started, elapsed_time
#     if is_timer_started:
#         # elapsed_time = int(time.time() - start_time)
#         elapsed_time = default_timer() - start_time
#         is_timer_started = False
#         print(f"Time taken to complete task => ({elapsed_time:.2f}s)")
#         # LOGGER.info(f"({elapsed_time:.2f}s)")
#         # LOGGER.info(f"Time taken to complete task => ({temp_end:.2f}s)")
#         return elapsed_time
#
#
# # Function to end program by changing some bool values
# def end_program():
#     global is_task_complete, is_start_path_tracking, is_tracker_start
#     is_task_complete = True
#     is_start_path_tracking = False
#     is_tracker_start = False
#     stop_timer()  # Record time taken to complete task
#     print("Total distance deviated from ideal path => ", overall_deviation)
#
#
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', draw_roi)
#
# print(
#     "[INFO] Click the left button: select the point, click the right button: delete the last selected point, click the middle button: confirm the ROI area")
# print("[INFO] Press 'S' to confirm the selected area and save")
# print("[INFO] Press 'Q' to stop tool tracking and calculate performance")
# print("[INFO]  ESC exit")
#
# while True:
#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:
#         break
#     if key == ord("s"):
#         saved_data = {
#             "ROI": pts
#         }
#         path_length = get_path_length()  # Calculate and retrieve total path length
#
#         print("The ROI coordinates have been saved locally..", saved_data, " \nTOTAL PATH LENGTH IS ", path_length)
#         break
# cv2.destroyAllWindows()


# print(pts)


# ********************** BEGIN PROGRAM *************************
# Open the Path Tracking Window
pk.start_task_path_tracker()


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    # Set Custom variables to Global global is_start_path_tracking, im0, is_task_complete, distance_travelled,
    # overall_deviation, x, y, poi, is_tracker_start, tooltip_dist, last_tracker_position, points_traversed

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # Set Custom variables to Global
    # global x, y, poi, is_tracker_start, tooltip_dist, last_tracker_position, points_traversed
    # ****************************************************************************************
    # *** MAIN PROGRAM LOOP ***
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            h, w, _ = im0.shape

            # ************** Write Current Task Iteration No. text on Screen **************
            font = cv2.FONT_HERSHEY_SIMPLEX
            # inserting text on video frame
            text_y_pos = int(0.03 * h)  # defining a y position for our text
            cv2.putText(im0, 'Current task ' + str(current_task_iteration + 1) + ' of ' + str(no_of_task_repeats),
                        (0, text_y_pos), font, 1, (255, 255, 255, 0.5),
                        3, cv2.LINE_4)
            # ------------------------------------------------

            if not pk.is_start_path_tracking:
                # ************** Write 'Start HERE' text on Screen **************
                font = cv2.FONT_HERSHEY_SIMPLEX
                # inserting text on video frame
                cv2.putText(im0, 'Start HERE', ((pk.pts[0][0] * w // 640), pk.pts[0][1] * h // 480), font, 1,
                            (255, 255, 255),
                            2, cv2.LINE_4)
                # ------------------------------------------------

            if len(pk.pts) > 1:
                cv2.circle(im0, (pk.pts[0][0] * w // 640, pk.pts[0][1] * h // 480), 9, (0, 255, 255),
                           7)  # draw first red circle // Start Point
                cv2.circle(im0, (pk.pts[-1][0] * w // 640, pk.pts[-1][1] * h // 480), 9, (0, 255, 0),
                           7)  # draw last blue circle // End Point
            for line in range(len(pk.pts)):
                try:

                    cv2.line(im0, (int(pk.pts[line][0] * w / 640), int(pk.pts[line][1] * h / 480)),
                             (int(pk.pts[line + 1][0] * w / 640), int(pk.pts[line + 1][1] * h / 480)), [255, 50, 10], 2,
                             cv2.LINE_AA)  # draw the path to be traversed created by the mouse
                except:
                    pass
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        # xyxy = torch.IntTensor(xyxy)

                        x1 = int(xyxy[0])
                        x1_r = int(640 / w * x1)
                        y1 = int(xyxy[1])
                        y1_r = int(480 / h * y1)
                        x2 = int(xyxy[2])
                        x2_r = int(640 / w * x2)
                        y2 = int(xyxy[3])
                        y2_r = int(480 / h * y1)
                        x = (x1 + (x1 + x2) // 2) // 2
                        x_r = (x1_r + (x1_r + x2_r) // 2) // 2
                        y = (y1 + y2) // 2
                        y_r = (y1_r + y2_r) // 2
                        point_of_interest = (x_r, y_r)  # Point on bbox/tooltip
                        # print("point_of_interest>>: ", point_of_interest)
                        print("update_variables>>>>>>>> ", pk.update_variables(index=0, t_value=(
                            int((x / w) * 640), int((y / h) * 480))))  # Update tooltip value
                        print("x, y = ", x, y)

                        # Draw Tracker at start position if not started
                        if not pk.is_tracker_start:
                            # last_tracker_position = tracker  # Store last tracker coordinates
                            # Draw tracker at Initial Start Position
                            pk.draw_tracker(im0, pk.pts[0][0], pk.pts[0][1], original_width=w, original_height=h)
                            # draw_tracker(im0, pts[0][0] * w // 640, pts[0][1] * h // 480)

                            # print("pts[0][0]====", pk.pts[0][0], "w====", w)
                            # print("pts[0][1]====", pk.pts[0][1], "h====", h)

                            pk.is_tracker_start = True  # Tracker drawn initially

                        else:
                            if pk.check_tracker_distance(pk.tracker) < pk.tracker_path_threshold and distance.euclidean(
                                    pk.tracker,
                                    pk.tool_tip) < pk.tracker_tooltip_threshold:  # Move tracker if close to path and tooltip by some val
                                # ********************************************************************
                                # Start tracking performance of path following
                                if not pk.is_start_path_tracking:
                                    # Start Timer
                                    pk.start_timer()
                                    pk.is_start_path_tracking = True

                                pk.last_tracker_position = pk.tracker  # Store last tracker coordinates while on path
                                # Store tracker movement coordinates
                                if pk.check_tracker_distance(
                                        pk.tracker) < pk.tracker_path_threshold:  # if tracker on path
                                    pk.points_traversed.append(pk.tracker)  # Store points travelled by tracker
                                    print("New Tracker Stored Coordinates .. ", pk.points_traversed)
                                #   *****  Draw Tracker for movement with tooltip   *****
                                pk.draw_tracker(im0, pk.tracker[0] * pk.tool_tip[0] // pk.tracker[0],
                                                pk.tracker[1] * pk.tool_tip[1] // pk.tracker[1], original_width=w,
                                                original_height=h)

                                print("Tracker tooltip DISTANCE //////// ", distance.euclidean(pk.tracker, pk.tool_tip))
                                print("Tracker MOVING")
                                # *********************************************************************
                                # Stop program if reached end of path
                                if pk.is_start_path_tracking:
                                    # End Program and Display Results
                                    if distance.euclidean(pk.tracker, pk.pts[-1]) < pk.tracker_end_point_threshold:
                                        print("Tracker reached end of path")
                                        pk.end_program()  # End Program
                                        print('Program ended normally')
                                        return
                            else:
                                # When tracker deviates from path no matter where tooltip is
                                if pk.check_tracker_distance(pk.tracker) > tracker_path_threshold:
                                    print("DEVIATION!!!!!!!!")
                                    # Store distances of deviation from ideal path
                                    # Introduce a temp variable to check whether new deviated point has changed before adding
                                    pk.overall_deviation += distance.euclidean(pk.tracker, pk.last_tracker_position)
                                    try:
                                        # Press a key to bring tracker to previous position on path
                                        if cv2.waitKey(1) & 0xFF == ord('b'):  # press 'b' key
                                            pk.draw_tracker(im0, pk.last_tracker_position[0], pk.last_tracker_position[1],
                                                            original_width=w, original_height=h)
                                            print("KEY 'b' PRESSED! TRACKER SUPPOSED TO RESET TO PREV LOC ON PATH")
                                            break
                                        else:
                                            # Draw motionless tracker if far from path  # This code was prev above 'try'
                                            print("Tracker MOTIONLESS")
                                            pk.draw_tracker(im0, pk.tracker[0], pk.tracker[1], original_width=w,
                                                            original_height=h)
                                    except:
                                        print('Error. Tracker Reset Key pressed at the wrong time.')

                        # print("poi (collected): ", poi)
                        print("Current tracker coordinates", pk.tracker, " tooltip ", pk.tool_tip)
                        # Check the distance between tooltip and tracker
                        pk.tooltip_dist = distance.euclidean(pk.tracker, pk.tool_tip)
                        print("Tooltip distance from tracker :::::::: ", pk.tooltip_dist)
                        # Draw tooltip small circle
                        pk.draw_tool_tip(im0, pk.tool_tip[0], pk.tool_tip[1], original_width=w,
                                         original_height=h)  # tooltip circle

                        # Exit program if Window closed cv2.getWindowProperty('window-name', 0) >= 0:
                        # if cv2.getWindowProperty(str(p), cv2.WND_PROP_VISIBLE) < 1:
                        #     print("Window closed")
                        #     sys.exit()

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        # Check the On_line Status of Tracker
        print("shortest distance ", pk.check_tracker_distance(pk.tracker))
        # Check the distance between tooltip and tracker
        # tooltip_dist = distance.euclidean(tracker, tool_tip)
        # print("Tooltip distance:::::::: ", tooltip_dist)
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Stream results
        try:
            im0 = annotator.result()
        except:
            pass
        # if view_img:
        im0 = cv2.resize(im0, (640, 480))
        cv2.imshow(str(p), im0)

        # Start Measure performance
        if cv2.waitKey(1) & 0xFF == 32:  # if space pressed
            print("Space pressed")
            if not pk.is_start_path_tracking:
                pk.is_start_path_tracking = True
                # break
        # Stop path following and show performance results
        # end_program()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # pk.stop_timer()
            pk.end_program()
            break

    # *******************************************************************
    # Calculate TOTAL DISTANCE TRAVELLED
    for n in range(len(pk.points_traversed)):
        if n < len(pk.points_traversed) - 1:
            pk.distance_travelled += distance.euclidean(pk.points_traversed[n], pk.points_traversed[n + 1])

    print("TOTAL DISTANCE TRAVERSED = ", pk.distance_travelled)

    # Get Performance Time
    if pk.is_timer_started:  # if timer still running for some reason
        print("Timer was still running")
        pk.end_program()
    print("Time taken to complete task ==> ", pk.elapsed_time, "s")

    # xs =[]
    # ys = []
    # for p in range(len(poi)):
    #     xs.append(abs(poi[p][0]))
    #     ys.append(abs(poi[p][1]-480))
    # # print(xs)
    # # print(ys)
    # plt.plot(xs, ys, marker='o', color='green', markerfacecolor='red', markersize=2)
    # # plt.plot(x[0], y[0], 'ro', color="green", markersize=6)
    # # plt.plot(x[-1], y[-1], 'ro', color="red", markersize=6)
    #
    # plt.ylim(0, 480)
    # plt.xlim(0, 640)
    # plt.title('Tool Tracking!')
    # # fig, ax = plt.subplots(1)
    # # for p in patches:
    # #     ax.add_patch(p)
    # plt.savefig('tracking_graph.png')
    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/david_best.pt', help='model path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'extracted.mp4', help='file/dir/URL/glob, 0 for webcam')
    # parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    global current_task_iteration
    while current_task_iteration < no_of_task_repeats:  # Specifying how many times to repeat task
        run(**vars(opt))
        current_task_iteration += 1


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
