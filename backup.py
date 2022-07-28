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
from datetime import timedelta
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
import asyncio
from tasks import path_tracking as pk
from gui import app
from gui import detect_marker

# Custom variables
no_of_task_repeats = app.repeat_task_value  # How many times to repeat the task in a session
current_task_iteration = 0
video_source = 0        # Indicate which source of video feed
is_screen_recording_started = False
pixel_cm_ratio = 0
is_pixel_cm_val_retrieved = False

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


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
    global video_source, pixel_cm_ratio, is_pixel_cm_val_retrieved

    source = str(source)
    video_source = source       # assign source of video feed to global custom variable
    # print(f"VIDEO SOURCE IS ......... {video_source}")
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

    # Data-loader
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

    # Create an async video writer func to call a method from another file
    async def run_async_video_writer():
        loop = asyncio.get_event_loop()
        loop.run_until_complete(app.write_to_video(video_source, './videos/recorded_tasks'))
        loop.close()
        # async_func = loop.create_task(app.write_to_video(video_source, './videos/recorded_tasks'))
        # app.write_to_video(video_source, './videos/recorded_tasks')

    # --------------------- Check Pixel-To-CM Ratio ----------------------
    if not is_pixel_cm_val_retrieved:   # run function once
        pixel_cm_ratio = detect_marker.detect_marker_in_background(
            "gui/detect_red3.MOV")  # default input=source as the same for main task
        is_pixel_cm_val_retrieved = True
        print("Pixel val retrieved ")

    # Set Custom variables to Global
    global is_screen_recording_started
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

        # SOme random loop count test. Not necessary for program running

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

        # Start time check for Loop
        start = default_timer()

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

            # ************************************ PATH TRACKING TASK HERE *********************************************
            # ************** Write Current Task Iteration No. text on Screen **************
            pk.display_current_task_iteration_no(im0, h, current_task_iteration, no_of_task_repeats)
            # ------------------------------------------------

            # ----------------------- Calculate Frames Per Sec (FPS) of video file ----------------------------
            app.get_frames_per_sec(video_source)


            if not pk.is_start_path_tracking:
                # ************** Write 'Start HERE' text on Screen **************
                pk.display_start_here_text(im0, (pk.pts[0][0]* w // 640), (pk.pts[0][1]* h // 480))
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
                        # c = int(cls)  # integer class
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # annotator.box_label(xyxy, label, color=colors(c, True))

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
                            # Draw tracker at Initial Start Position
                            pk.draw_tracker(im0, pk.pts[0][0], pk.pts[0][1], original_width=w, original_height=h)

                            pk.is_tracker_start = True  # Tracker drawn initially
                        else:
                            if pk.check_tracker_distance(pk.tracker) < pk.tracker_path_threshold and distance.euclidean(
                                    pk.tracker,
                                    pk.tool_tip) < pk.tracker_tooltip_threshold:  # Move tracker if close to path & tooltip by some val
                                # ********************************************************************
                                # Start tracking performance of path following
                                if not pk.is_start_path_tracking:
                                    # Start Timer
                                    pk.start_timer()
                                    # Start Recording Screen
                                    if not is_screen_recording_started:
                                        # asyncio.run(run_async_video_writer())    # run asynchronously
                                        is_screen_recording_started = True
                                    pk.is_start_path_tracking = True

                                # Record all coordinates traversed while path tracking and timer started
                                if pk.is_start_path_tracking:
                                    pk.all_coordinates_recorded.append(pk.tracker)

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
                                        print("Pixel cm value in Detect file is ", pixel_cm_ratio)
                                        pk.end_program(pixel_cm_ratio)  # ******************** End Task Training Program ************ 58.5872431945801
                                        # ...and dispatch pixel_cm_ratio val for storage
                                        # ********************* SAVE results to file if all iterations complete *******
                                        if no_of_task_repeats - current_task_iteration == 1:
                                            pk.save_results()
                                            print('All Iteration Results saved to file')
                                        print('Program ended normally')
                                        is_screen_recording_started = False     # Screen recording ended
                                        return
                            else:   # if task started BUT DEVIATED
                                # When tracker deviates from path no matter where tooltip is
                                if pk.check_tracker_distance(pk.tracker) >= pk.tracker_path_threshold:
                                    print("DEVIATION!!!!!!!!")
                                    # FREEZE tracker while off-path
                                    print("Tracker MOTIONLESS")
                                    pk.draw_tracker(im0, pk.tracker[0], pk.tracker[1], original_width=w, original_height=h)
                                    # Reset Tracker to last optimal point on ideal path
                                    if not pk.is_tracker_resetting:
                                        # pk.write_some_text(im0, w, h, "You moved away from path. Please wait...")
                                        pk.center_text(im0, "You moved away from path. Please wait...")
                                        pk.reset_tracker_on_timer(2, im0)  # Wait for 2 secs before execution
                                    # ***** Tracker reset complete *****

                        # print("poi (collected): ", poi)
                        print("Current tracker coordinates", pk.tracker, " tooltip ", pk.tool_tip)
                        # Check the distance between tooltip and tracker
                        pk.tooltip_dist = distance.euclidean(pk.tracker, pk.tool_tip)
                        print("Tooltip distance from tracker :::::::: ", pk.tooltip_dist)
                        # Draw tooltip small circle
                        pk.draw_tool_tip(im0, pk.tool_tip[0], pk.tool_tip[1], original_width=w,
                                         original_height=h)  # tooltip circle

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
        # End time check for loop
        end = (default_timer() - start)
        LOGGER.info(f"!!TIME - Predictions Loop ::==== {timedelta(seconds=end)}")

        # Check the On_line Status of Tracker
        print("shortest distance ", pk.check_tracker_distance(pk.tracker))

        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        # Stream results
        try:
            im0 = annotator.result()
        except:
            pass
        # if view_img:        # Worked well while commented out. Uncommneted it out for just testing
        im0 = cv2.resize(im0, (640, 480))
        cv2.imshow(str(p), im0)     # display main task training window

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
            pk.end_program(pixel_cm_ratio)
            break
        if cv2.getWindowProperty(str(p), cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed from x button")
            sys.exit()

    # *******************************************************************
    # Calculate TOTAL DISTANCE TRAVELLED
    for n in range(len(pk.points_traversed)):
        if n < len(pk.points_traversed) - 1:
            pk.distance_travelled += distance.euclidean(pk.points_traversed[n], pk.points_traversed[n + 1])

    print("TOTAL DISTANCE TRAVERSED = ", pk.distance_travelled)
    print("ALL COORDINATES RECORDED = ", pk.all_coordinates_recorded)

    # Get Performance Time
    if pk.is_timer_started:  # if timer still running for some reason
        print("Timer was still running")
        pk.end_program(pixel_cm_ratio)
    print("Time taken to complete task ==> ", pk.elapsed_time, "s")

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
