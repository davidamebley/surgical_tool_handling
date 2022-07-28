"""
This contains the path tracking task
"""

import argparse
import time
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
import threading
from gui import app

# Check if path tracking started
is_start_path_tracking = False

# Check if timer started
is_timer_started = False

# Check if task complete
is_task_complete = False

# Check if tracker has started moving
is_tracker_start = False

# Other Custom Variables
x, y, im0 = 0, 0, 0
# Custom variables
path_length = 0
tracker_path_threshold = 20  # 10 20
tracker_tooltip_threshold = 100  # 10 300
tracker_end_point_threshold = 8  # 8 is min. 300
poi = []
tool_tip = []
tracker = []
tooltip_dist = 0
last_tracker_position = []
points_traversed = []
all_coordinates_recorded = []
distance_travelled = 0
elapsed_time = 0  # time taken to complete task
start_time = 0  # time task started
overall_deviation = 0  # Deviation from ideal path (in distance)
is_tracker_resetting = False  # Check if tracker being reset
path_saved_coord_columns = ['path_id', 'coord_x', 'coord_y']
saved_traversed_coord_columns = ['id', 'coord_x', 'coord_y']  # columns for actual saved coordinates
all_traversed_coord_columns = ['id']  # columns for how many coordinates saved or ID
path_saved_id_columns = ['path_id', 'length']
last_path_id = 0
last_traversed_coordinates_id = 0

# pts = []
# Custom points for a default line. Can be removed by commenting it out
# Path coordinates defined
pts = app.predefined_training_path_coordinates
# pts = [(424, 305), (418, 329), (408, 346), (405, 326), (405, 300), (405, 274), (408, 246), (408, 222), (407, 196),
#        (415, 163), (404, 136), (401, 109), (386, 86), (379, 64), (365, 87), (358, 108), (347, 131), (329, 156),
#        (318, 186), (307, 208), (295, 231), (287, 256), (281, 284), (276, 311), (271, 340), (269, 315), (246, 299),
#        (233, 272), (223, 250), (207, 228), (199, 202), (194, 169), (183, 142), (170, 115), (164, 92), (161, 63)]

# pts2 = [(397, 238), (428, 248), (450, 274), (446, 291), (433, 309), (414, 323), (389, 334), (358, 335), (327, 320),
# (301, 298), (286, 265), (295, 224), (321, 197), (347, 184), (378, 178), (414, 177), (446, 183), (473, 198), (486,
# 218), (503, 252), (521, 278), (527, 307), (523, 331), (503, 349), (481, 366), (443, 380), (402, 392), (362, 394),
# (344, 396), (323, 396), (301, 393), (260, 373), (240, 350), (230, 325), (216, 288), (208, 251), (208, 209), (216,
# 178), (243, 146), (283, 131), (333, 123), (376, 116), (420, 111), (465, 116), (506, 139), (528, 166), (547, 183),
# (565, 216), (579, 256), (585, 299), (593, 341), (594, 383), (543, 420), (513, 445), (448, 456), (396, 459), (338,
# 461), (296, 462), (243, 446), (203, 430), (186, 404), (168, 362), (149, 325), (149, 282), (146, 248), (138, 217),
# (153, 185), (176, 140), (196, 119), (228, 89), (241, 70), (237, 43), (221, 26), (185, 23), (154, 23), (121, 35),
# (108, 56), (95, 71), (87, 87), (73, 114), (60, 128), (43, 151), (34, 182), (24, 207), (19, 244), (24, 281), (24,
# 307), (35, 335), (36, 361), (25, 379), (21, 406), (55, 440)]

img_poly = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
# img_poly = np.zeros(shape=[960, 1280, 3], dtype=np.uint8)


# Draw path with mouse
def draw_roi(event, x, y, flags, param):
    img2 = img_poly.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        # print(x,y)
    if event == cv2.EVENT_RBUTTONDOWN:
        pts.pop()

    if event == cv2.EVENT_MBUTTONDOWN:
        mask = np.zeros(img_poly.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))

        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))

        show_image = cv2.addWeighted(src1=img_poly, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

        cv2.imshow("mask", mask2)
        cv2.imshow("show_img", show_image)

        ROI = cv2.bitwise_and(mask2, img_poly)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)

    if len(pts) > 0:
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:

        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow('image', img2)
    # return pts, cv2.imshow('image', img2)
    return pts


def draw_tracker(im, point_x, point_y, original_width=1920, original_height=1080):
    # Draw Tracker circle
    cv2.circle(im, (point_x * original_width // 640, point_y * original_height // 480), 7, (0, 0, 255), 7)
    # pts[0][0] * w // 640, pts[0][1] * h // 480
    update_variables(index=1, t_value=(point_x, point_y))


def draw_tool_tip(im, point_x, point_y, original_width=1920, original_height=1080):
    # Draw Tracker circle
    cv2.circle(im, (point_x * original_width // 640, point_y * original_height // 480), 3, [0, 255, 255], 5)
    # int((x / w) * 640), int((y / h) * 480)
    update_variables(index=0, t_value=(point_x // original_width * 640, point_y // original_height * 480))


# Update My Tracking Variables
def update_variables(index, t_value):
    global tool_tip, tracker
    if index == 0:
        tool_tip = t_value
    elif index == 1:
        tracker = t_value
    return t_value


# Function to determine the on_line status of tracker
def is_tracker_on_line(m_tracker):
    a, temp = 0, 0
    for n in range(len(pts)):
        if n < len(pts) - 1:
            coord_x1, coord_y1 = pts[n]  # Current point
            coord_x2, coord_y2 = pts[n + 1]  # Next point
            dist = abs((coord_y2 - coord_y1) * m_tracker[0] - (coord_x2 - coord_x1) * m_tracker[
                1] + coord_x2 * coord_y1 - coord_y2 * coord_x1) / math.sqrt(
                (coord_y2 - coord_y1) ** 2 + (coord_x2 - coord_x1) ** 2)
            if n == 0:
                a = dist
            else:
                temp = dist
                if temp < a:
                    a = temp
            print("current distance = ", dist)
    return a


# Function to check the shortest distance between tracker and drawn path
def check_tracker_distance(m_tracker):
    try:
        trk_x = m_tracker[0]
        trk_y = m_tracker[1]
        short, temp = 0, 0
        for n in range(len(pts)):
            if n < len(pts) - 1:
                x1, y1 = pts[n]
                x2, y2 = pts[n + 1]
                a = trk_x - x1
                b = trk_y - y1
                c = x2 - x1
                d = y2 - y1

                dot = a * c + b * d
                len_sq = c * c + d * d
                param = -1

                if len_sq != 0:  # in case of 0 length line
                    param = dot / len_sq

                if param < 0:
                    xx = x1
                    yy = y1
                elif param > 1:
                    xx = x2
                    yy = y2
                else:
                    xx = x1 + param * c
                    yy = y1 + param * d

                dx = trk_x - xx
                dy = trk_y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                # Store the shortest distance
                if n < 1:
                    short = dist
                else:
                    temp = dist
                    if temp < short:
                        short = temp  # shortest distance
                print("distance at ", pts[n], " and ", pts[n + 1], "===:: ", dist, " .shortest dist is ", short)
        return short
    except:
        print("ERROR in checking tracker distance to path")


# Function to generate total length of drawn path
def get_path_length():
    length = 0
    try:
        for n in range(len(pts)):
            if n < len(pts) - 1:
                length += distance.euclidean(pts[n], pts[n + 1])
        return length
    except:
        print("invalid path length")


# Function to retrieve last saved path id
def get_last_traversed_coord_id():
    global last_traversed_coordinates_id
    raw_path_data = get_traversed_coord_id_csv()

    d_f = raw_path_data
    # top = d_f.head(1)
    try:
        last_row = d_f.tail(1)
    except AttributeError:
        return 0  # return value zero if csv file returns nothing
    n_bottom = 0
    for n_row in last_row.itertuples(index=False):
        n_bottom = getattr(n_row, "id")  # getting the 'id' part of the row
    last_traversed_coordinates_id = n_bottom
    return n_bottom


# Create a unique name for traversed path coordinates before saving
def namify_traversed_path_coord(coord_list, last_id, accumulator=1):
    new_id = int(last_id) + accumulator  # add val 1 to the last id retrieved to create a new id
    n_user = 1
    path_id = 1
    n_iter = 0
    temp_list = []
    stored_list = []

    new_list = []
    for i, sublist in enumerate(coord_list):
        print(sublist)
        for n_item in sublist:
            # Criteria for naming: user_ path_id_ iteration_ saved_coordinates_id_ YYYYHMS_ then coordinates' x and y
            coord_id = str(n_user) + str(path_id) + str(i+1) + str(new_id) + dummy_times[i]
            temp_list.append(coord_id)
            coord_xy = ','.join(map(str, n_item))
            coord_xy = coord_xy.split(',')
            [temp_list.append(n_item) for n_item in coord_xy]
            tupled_list = tuple([int(float(n_item)) for n_item in temp_list])
            stored_list.append(tupled_list)
            temp_list = []
        new_list = stored_list
    return new_list


# Function to save/export Path Tracking task all traversed coordinates to files
def save_traversed_path_coordinates(path_coord_list):
    # Export data to external file
    global last_traversed_coordinates_id
    # Header Columns
    columns_traversed_coord = saved_traversed_coord_columns
    columns_coord_id = all_traversed_coord_columns
    # Get last Path id
    last_coord_id = get_last_traversed_coord_id()
    coordinates_data = namify_traversed_path_coord(path_coord_list, last_coord_id)
    # print("printing", path_data)
    print(coordinates_data)
    # return
    dt_frame = pd.DataFrame(coordinates_data)
    new_path_coord_id = last_coord_id + 1
    path_coord_id_data = [(new_path_coord_id,)]  # Add val 1 to last id to create a new one
    # print("printing2", path_id_data)
    dt_frame_path_coord_id = pd.DataFrame(path_coord_id_data)
    file_name_path_coord = "./extra_files/saved_traversed_coordinates.csv"
    file_name_path_ccord_id = "./extra_files/all_traversed_coordinates.csv"
    try:
        # Save path coordinates
        if not os.path.isfile(file_name_path_coord):
            with open(file_name_path_coord, 'a', newline='') as f:
                print("New file created")
                dt_frame.to_csv(f, header=columns_traversed_coord, index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)

        else:
            with open(file_name_path_coord, 'a', newline='') as f:
                dt_frame.to_csv(f, header=(f.tell() == 0), index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)

        # Save path id to file
        # ------------------------------------------
        if not os.path.isfile(file_name_path_ccord_id):
            with open(file_name_path_ccord_id, 'a', newline='') as f:
                print("New file created")
                dt_frame_path_coord_id.to_csv(f, header=columns_coord_id, index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)

        else:
            with open(file_name_path_ccord_id, 'a', newline='') as f:
                dt_frame_path_coord_id.to_csv(f, header=(f.tell() == 0), index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)

    except OSError as e:
        raise OSError(e)


# Function to save/export Path Tracking task new path details to files
def save_predefined_path_coordinates(path_coord_list):
    # Export data to external file
    global last_path_id
    # Header Columns
    columns_coord = path_saved_coord_columns
    columns_id = path_saved_id_columns
    # Get last Path id
    last_path_id = get_last_path_id()
    path_data = prepend_id_to_path_coord(path_coord_list, last_path_id)
    # print("printing", path_data)
    dt_frame = pd.DataFrame(path_data)
    new_path_id = last_path_id + 1
    path_id_data = [(new_path_id, path_length,)]  # Add val 1 to last id to create a new one
    # print("printing2", path_id_data)
    dt_frame_path_id = pd.DataFrame(path_id_data)
    file_name_path_coord = "./extra_files/saved_path_coordinates.csv"
    file_name_path_id = "./extra_files/training_paths.csv"
    try:
        # Save path coordinates
        if not os.path.isfile(file_name_path_coord):
            with open(file_name_path_coord, 'a', newline='') as f:
                print("New file created")
                dt_frame.to_csv(f, header=columns_coord, index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)

        else:
            with open(file_name_path_coord, 'a', newline='') as f:
                dt_frame.to_csv(f, header=(f.tell() == 0), index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)

        # Save path id to file
        # ------------------------------------------
        if not os.path.isfile(file_name_path_id):
            with open(file_name_path_id, 'a', newline='') as f:
                print("New file created")
                dt_frame_path_id.to_csv(f, header=columns_id, index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)

        else:
            with open(file_name_path_id, 'a', newline='') as f:
                dt_frame_path_id.to_csv(f, header=(f.tell() == 0), index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)
        # Update Selected Path ID to that of Newly created path
        app.selected_predefined_training_path = new_path_id
        print("New path id is ", str(app.selected_predefined_training_path))

    except FileNotFoundError:
        raise FileNotFoundError('Error writing to file')


# Prepend id to newly created path coordinates before saving
def prepend_id_to_path_coord(list, last_id, accumulator=1):
    path_id = (int(last_id) + accumulator,)  # add val 1 to the last id retrieved to create a new id
    new_list = []
    for sublist in list:
        new_list.append(path_id + sublist)
    return new_list


# Function to retrieve last saved path id
def get_last_path_id():
    global last_path_id
    raw_path_data = get_path_csv()

    d_f = raw_path_data
    # top = d_f.head(1)
    try:
        last_row = d_f.tail(1)
    except AttributeError:
        return 0  # return value zero if csv file returns nothing
    n_bottom = 0
    for n_row in last_row.itertuples(index=False):
        n_bottom = getattr(n_row, "path_id")
    last_path_id = n_bottom
    return n_bottom


# Function to get the csv file of Path ID file
def get_path_csv():
    # Retrieve path coordinates data from file ********************************************
    try:
        raw_path_data = pd.read_csv("./extra_files/training_paths.csv", names=path_saved_id_columns, header=0)
        return raw_path_data
    except OSError as e:
        raise FileNotFoundError("error ", e)


# Function to get the csv file of Path Coordinates file
def get_path_coordinates_csv():
    # Retrieve path coordinates data from file ********************************************
    try:
        raw_path_data = pd.read_csv("./extra_files/saved_path_coordinates.csv", names=path_saved_coord_columns,
                                    header=0)
        return raw_path_data
    except OSError as e:
        raise FileNotFoundError("error ", e)


# Backup to function to reset tracker to ideal path position
# def reset_tracker(frame):
#     global overall_deviation, is_tracker_resetting
#     # Store distances of deviation from ideal path
#     overall_deviation += distance.euclidean(tracker, last_tracker_position)
#     # try:
#     # Bring tracker to ideal previous position on path
#     draw_tracker(frame, last_tracker_position[0], last_tracker_position[1], original_width=1920, original_height=1080)
#     is_tracker_resetting = False
#     print("Tracker reset successfully")


# Function to reset tracker to ideal path position
def reset_tracker(frame):
    global is_tracker_resetting
    # Reset tracker to ideal previous position on path
    is_tracker_resetting = False
    draw_tracker(frame, last_tracker_position[0], last_tracker_position[1], original_width=1920, original_height=1080)
    print("Tracker reset successfully")


# Back up to function to run a timed function
# def reset_tracker_on_timer(secs, tracker_frame):
#     global is_tracker_resetting
#     timer_start_time = threading.Timer(secs, lambda: reset_tracker(tracker_frame))      # reset tracker after some secs
#     timer_start_time.start()
#     is_tracker_resetting = True
#     print("Tracker initiating reset...")

def reset_tracker_on_timer(secs, tracker_frame):
    global is_tracker_resetting, overall_deviation
    is_tracker_resetting = True
    print("Tracker resetting...")
    # draw_tracker(tracker_frame, tracker[0], tracker[1])         # FREEZE tracker
    # Store distances of deviation from ideal path
    overall_deviation += distance.euclidean(tracker, last_tracker_position)
    print("Deviation distance stored")
    timer_start_time = threading.Timer(secs, lambda: reset_tracker(tracker_frame))      # reset tracker after countdown
    timer_start_time.start()        # start countdown timer


# Function to Write Current Task Iteration No. text on Screen
def display_current_task_iteration_no(frame, frame_height, current_iteration, no_of_iterations):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # inserting text on video frame
    text_y_pos = int(0.03 * frame_height)  # defining a y position for our text
    cv2.putText(frame, 'Current task ' + str(current_iteration + 1) + ' of ' + str(no_of_iterations),
                (0, text_y_pos), font, 1, (255, 255, 255, 0.5), 3, cv2.LINE_4)

# Write some text on screen
def write_some_text(frame, frame_width, frame_height, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # inserting text on frame
    x_pos = int(0.5 * frame_width)  # defining a y position for our text
    y_pos = int(0.5 * frame_height)  # defining a y position for our text
    cv2.putText(frame, text, (x_pos, y_pos), font, 1, (255, 255, 255, 0.5), 3, cv2.LINE_4)


# font = cv2.FONT_HERSHEY_SIMPLEX
#                 # inserting text on video frame
#                 cv2.putText(im0, 'Start HERE', ((pk.pts[0][0] * w // 640), pk.pts[0][1] * h // 480), font, 1,
#                             (255, 255, 255),
#                             2, cv2.LINE_4)


# Function to center some text on screen
def center_text(img, text):
    # get boundary of this text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 3)[0]

    # get coords based on boundary
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2

    # add text centered on image frame
    cv2.putText(img, text, (text_x, text_y), font, 1, (0, 0, 200), 2)


def display_start_here_text(img, x_size, y_size):
    # get boundary of this text
    text = "Start HERE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 3)[0]

    # get coords based on boundary
    # text_x = (img.shape[1] - text_size[0]) // 2
    # text_y = (img.shape[0] + text_size[1]) // 2
    text_x = (x_size - text_size[0])
    text_y = (y_size + text_size[1])

    # add text centered on image frame
    cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 2)


# Function to start timing performance
def start_timer():
    global is_timer_started, start_time
    if not is_timer_started:
        start_time = time.time()
        # start_time = default_timer()  # default_timer is independent of unfortunate system time's changes during usage
        is_timer_started = True
        return start_time


# Function to stop performance timer
def stop_timer():
    global is_timer_started, elapsed_time
    if is_timer_started:
        elapsed_time = int(time.time() - start_time)
        # elapsed_time = default_timer() - start_time
        is_timer_started = False
        print(f"Time taken to complete task => ({elapsed_time:.2f}s)")
        # LOGGER.info(f"({elapsed_time:.2f}s)")
        # LOGGER.info(f"Time taken to complete task => ({temp_end:.2f}s)")
        return elapsed_time


# Function to calculate Total Distance Traversed
def get_total_distance_traversed():
    # Calculate TOTAL DISTANCE TRAVELLED
    global distance_travelled, points_traversed
    distance_travelled = 0      # rest global distance travelled value
    for n in range(len(points_traversed)):
        if n < len(points_traversed) - 1:
            distance_travelled += distance.euclidean(points_traversed[n], points_traversed[n + 1])
            print("distance_travelled +=: ", str(distance_travelled))

    print("TOTAL DISTANCE TRAVERSED =   ", distance_travelled)
    print("Total length of points traversed array: ", str(len(points_traversed)))
    del points_traversed[:]       # Reset array after each training iteration
    print("Total length of points traversed array after Del: ", str(len(points_traversed)))
    return distance_travelled


# Function to Round list items
def round_list_items(n_list, val):
    results = [round(curr_item, val) for curr_item in n_list]
    return results


# Function to end program and keep results temporarily
def end_program(pixel_cm_ratio):
    global is_task_complete, is_start_path_tracking, is_tracker_start, overall_deviation
    is_task_complete = True
    is_start_path_tracking = False
    is_tracker_start = False
    stop_timer()  # Record time taken to complete task
    get_total_distance_traversed()  # Distance traveled by tracker
    print("Total distance deviated from ideal path => ", overall_deviation)
    print("Pixel ratio => ", pixel_cm_ratio)

    # Prepare results for subsequent file storage
    results = round_list_items(
        [elapsed_time, distance_travelled, overall_deviation, app.selected_predefined_training_path, pixel_cm_ratio],
        1)  # 'duration', 'distance_traversed', 'deviation', 'training_path', (round val)
    overall_deviation = 0  # reset global deviation val
    app.append_results(results)  # Temp store results in a list


# Function to save results to file
def save_results():
    app.save_results()


def start_task_path_tracker():
    global path_length
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)

    while True:
        key = cv2.waitKey(33) & 0xFF
        # if key == 27:  # Esc pressed
        #     break
        if key == 32:  # Space pressed      Proceed without saving path coordinates
            break
        if key == ord("s"):  # 'S' Pressed. Save coordinates and proceed to training
            if app.is_drawn_new_path:
                saved_data = {
                    "ROI": pts
                }
                # save path coordinates to file
                save_predefined_path_coordinates(pts)
                path_length = get_path_length()  # Calculate and retrieve total path length
                print("The ROI coordinates have been saved locally..", saved_data, " \nTOTAL PATH LENGTH IS ",
                      path_length)
                break
        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed")
            sys.exit()
    cv2.destroyAllWindows()
