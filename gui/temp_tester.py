import pandas as pd
import os
import threading
from PIL import ImageFont
import numpy as np
import cv2
from PIL import ImageGrab

pts = [(424, 305), (418, 329), (408, 346), (405, 326), (405, 300), (405, 274), (408, 246), (408, 222), (407, 196),
       (415, 163), (404, 136), (401, 109), (386, 86), (379, 64), (365, 87), (358, 108), (347, 131), (329, 156),
       (318, 186), (307, 208), (295, 231), (287, 256), (281, 284), (276, 311), (271, 340), (269, 315), (246, 299),
       (233, 272), (223, 250), (207, 228), (199, 202), (194, 169), (183, 142), (170, 115), (164, 92), (161, 63)]

pts2 = [(100, 140), (0, 200), (300, 600), (200, 0), (400, 400), (400, 100)]
dummy_times = ['2022192803', '2022192853']

# raw = [['david', 40, 53, 71], ['david', 17, 91, 200], ['david', 30, 81, 97], ['sammy', 23, 44, 36],
#        ['sammy', 62, 52, 21]]
#
# columns = ['username', 'time', 'distance', 'deviation']
# dt_frame = pd.DataFrame(raw)
# file_name = '../extra_files/results.csv'
# try:
#     if not os.path.isfile(file_name):
#         with open(file_name, 'a', newline='') as f:
#             print("No file initially")
#             dt_frame.to_csv(f, header=columns, index=False, mode='a')
#     else:
#         with open(file_name, 'a', newline='') as f:
#             dt_frame.to_csv(f, header=(f.tell() == 0), index=False, mode='a')
# except e:
#     print(e)

# *************************************
# skips the passed rows in new series
# try:
csv_data = pd.read_csv("../extra_files/results.csv", names=['username', 'duration', 'distance_traversed', 'deviation'])
csv_data2 = pd.read_csv("../extra_files/users.csv", names=['username'], header=0)
df = pd.DataFrame(csv_data, columns=['username', 'duration', 'distance_traversed', 'deviation'])
performance_results_columns = ['username', 'duration', 'distance_traversed', 'deviation']
path_saved_coord_columns = ['path_id', 'coord_x', 'coord_y']
path_saved_id_columns = ['path_id', 'length']
saved_traversed_coord_columns = ['id', 'coord_x', 'coord_y']  # columns for actual saved coordinates
all_traversed_coord_columns = ['id']  # columns for how many coordinates saved or ID
last_traversed_coordinates_id = 0
new_data = list(csv_data)
last_path_id = 0

# ******************************************************************************
for item in new_data:
    print(csv_data[item][5])
    # df[row][2]
# except FileNotFoundError:
#     print("Error while reading file")
# for row in csv_data.index:
#     print(row['username'])
print("*******************")

som = []
for row in csv_data2.username:
    som.append(row)
print(som)

print("*******************")

something_list = []
something = []


# for row in csv_data.itertuples(index=False):
#     # for index, row in csv_data.iterrows():
#     if row.username == "mmm":
#         for sub_i in row:
#             if not isinstance(sub_i, (str, bytes)):
#                 something_list.append(float(sub_i))
#             else:
#                 something_list.append(sub_i)
#         something.append(something_list)
# something = (row.username, row.duration, row.distance_traversed, row.deviation)


# Get user results from file
def get_results(username):
    is_user_present = False
    result_list = []
    for row in csv_data.itertuples(index=False):
        if row.username == username:
            is_user_present = True
            some_attr = getattr(row, test_something())
            result = [getattr(row, test_something()), float(row.duration), float(row.distance_traversed),
                      float(row.deviation)]
            result_list.append(result)
    if not is_user_present:
        print("No result found")
        return
    print("Printing getattr")
    return result_list


def test_something():
    return "username"


ind = 3
for val in range(len(performance_results_columns)):
    if ind == val:
        print("index")
        print(performance_results_columns[ind])


def fun(name):  # user defined function which adds +10 to given number
    print(name + ", Hey u called me")
    print("Done calling")


def run_on_timer(secs, function):
    timer_start_time = threading.Timer(secs, function)
    timer_start_time.start()
    print("Time started")


def run_my_timer(secs):
    timer_start_time = threading.Timer(secs, lambda: fun("dave"))
    timer_start_time.start()
    print("Time started")


def get_pil_text_size(text, font_size, font_name):
    font = ImageFont.truetype(font_name, font_size)
    size = font.getsize(text)
    print(size)


def center_text(img, text):
    # get boundary of this text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 3)[0]

    # get coords based on boundary
    text_x = (img.shape[1] - text_size[0]) / 2
    text_y = (img.shape[0] + text_size[1]) / 2

    # add text centered on image frame
    cv2.putText(img, text, (text_x, text_y), font, 1, (255, 255, 255), 3)


def get_boundary_points():
    # lx, ty, rx, by = 0, 0, 0, 0 (424, 305), (418, 329), (408, 346)
    rx, by = 0, 0
    lx, ty = pts[0][0], pts[0][1]
    for point in pts:
        if point[0] < lx:
            lx = point[0]
        if point[1] < ty:
            ty = point[1]
        if point[0] > rx:
            rx = point[0]
        if point[1] > by:
            by = point[1]
    final_pt = lx, ty, rx, by
    return final_pt


# def not_to_be_run():
#     # **************************************************
#     # Drawing Lines
#     if len(pts) > 1:
#
#         for i in range(len(pts) - 1):
#             cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
#             cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
#
#     # cv2.imshow('image', img2)
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()
#     return pts, cv2.imshow('image', img2)
#     # ************************************* End of lines


# def try_screen_capture():
#     img_poly = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
#     img2 = img_poly.copy()
#     if len(pts) > 1:
#         for i in range(len(pts) - 1):
#             cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
#             cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
#     cv2.imshow('image', img2)
#     # cv2.imwrite("saved_image.png", image)
#     cv2.waitKey()
#     return pts, cv2.imshow('image', img2)


# def try_screen_capture():
# # img = ImageGrab.grab(bbox=(0, 1000, 100, 1100))  # x, y, w, h
# img = ImageGrab.grab(bbox=(get_boundary_points()))  # x, y, w, h
# img_np = np.array(img)
# frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
# # Trying some window for our position tracking
# img_poly = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
# img2 = img_poly.copy()
# cv2.namedWindow('image')
# while True:
#     cv2.imshow("frame", frame)
#     pos = cv2.getWindowImageRect("image")  # Get window pos coordinates
#     print(f"Window position is {pos}")
#     if cv2.waitKey(1) & 0Xff == ord('q'):
#         break
# cv2.destroyAllWindows()


# def try_window_pos():
#     img_poly = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
#     img2 = img_poly.copy()
#     cv2.namedWindow('image')
#     cv2.imshow('image', img2)
#     # *****************************
#     # img = cv2.imread("test.png")
#     # winname = "Test"
#     # cv2.namedWindow(winname)  # Create a named window
#     # cv2.moveWindow(winname, 40, 30)  # Move it to (40,30)
#     # cv2.imshow(winname, img)
#     cv2.waitKey()
#     cv2.destroyAllWindows()


def draw_lines():
    img_poly = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
    img2 = img_poly.copy()
    cv2.namedWindow('image')
    if len(pts) > 1:

        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow('image', img2)
    cv2.waitKey()


# Function to save/export Path Tracking task new path details to files
def save_predefined_path_coordinates(pts2):
    # Export data to external file
    global last_path_id
    path_length = 0
    # Header Columns
    columns_coord = path_saved_coord_columns
    columns_id = path_saved_id_columns
    # Get last Path id
    last_path_id = get_last_path_id()
    path_data = prepend_id_to_path_coord(pts2, last_path_id)
    print("printing", path_data)
    dt_frame = pd.DataFrame(path_data)
    path_id_data = [(last_path_id + 1, path_length,)]  # Add val 1 to last id to create a new on
    print("printing2", path_id_data)
    dt_frame_path_id = pd.DataFrame(path_id_data)
    file_name_path_coord = "../extra_files/saved_path_coordinates.csv"
    file_name_path_id = "../extra_files/training_paths.csv"
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

    except FileNotFoundError:
        raise FileNotFoundError('Error writing to file')


# Prepend id to newly created path coordinates before saving
def prepend_id_to_path_coord(list, last_id, accumulator=1):
    path_id = (int(last_id) + accumulator,)  # add val 1 to the last id retrieved to create a new id
    new_list = []
    for sublist in list:
        new_list.append(path_id + sublist)
    return new_list


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
        raw_path_data = pd.read_csv("../extra_files/training_paths.csv", names=path_saved_id_columns, header=0)
        return raw_path_data
    except OSError as e:
        raise FileNotFoundError("error ", e)


# Function to get the csv file of Traversed coord ID file
def get_traversed_coord_id_csv():
    # Retrieve path coordinates data from file ********************************************
    try:
        raw_path_data = pd.read_csv("../extra_files/all_traversed_coordinates.csv", names=all_traversed_coord_columns,
                                    header=0)
        return raw_path_data
    except OSError as e:
        raise FileNotFoundError("error ", e)


# Function to get the csv file of Path Coordinates file
def get_path_coordinates_csv():
    # Retrieve path coordinates data from file ********************************************
    try:
        raw_path_data = pd.read_csv("../extra_files/saved_path_coordinates.csv", names=path_saved_coord_columns,
                                    header=0)
        return raw_path_data
    except OSError as e:
        raise FileNotFoundError("error ", e)


# print(get_results("pp"), len(get_results("pp")))
# new_list = get_results("pp")
# for i, list in enumerate(new_list):
#     print(i, list)

# def run_my_timer2(secs, func):
#     timer_start_time = threading.Timer(secs, lambda: fun("dave"))
#     timer_start_time.start()
#     print("Time started")


# ******************** Graph Stuff 1 *************************
#     # the figure that will contain the plot
#     fig = Figure(figsize=(10, 10), dpi=100)
#
#     # list of results
#     x = [x + 1 for x in range(len(result_list))]
#     y = result_list
#
#     # adding the subplot
#     plot1 = fig.add_subplot(111)
#
#     # plotting the graph
#     plot1.plot(x, y, marker="o")
#     # plot1.legend()
#     plot1.set_xlabel("No. of times trained")
#     plot1.set_ylabel(label)
#
#     # df2.plot(kind='line', legend=True, ax=ax2, color='b', marker='o', fontsize=10)
#     plot1.set_title(f"User: [{username}] \n{label} for {selected_path}")
#
#     print(result_list)
#
#     # creating the Tkinter canvas
#     # containing the Matplotlib figure
#     canvas = FigureCanvasTkAgg(fig, master=results_window)
#     canvas.draw()
#
#     # placing the canvas on the Tkinter window
#     canvas.get_tk_widget().pack()
#
#     # creating the Matplotlib toolbar
#     toolbar = NavigationToolbar2Tk(canvas, results_window)
#     toolbar.update()
#
#     # placing the toolbar on the Tkinter window
#     canvas.get_tk_widget().pack()
#
#     # Add this open window to a list
#     current_open_results_window_list.append(results_window)


# delay = int(input("Enter the delay time :"))
# start_time = threading.Timer(delay, lambda: fun("dave"))
# start_time.start()
# print("Time started")

# run_my_timer(5)

# print(f"Four boundary coordinates of points are {get_boundary_points()}")
# try_window_pos()
# try_screen_capture()
# draw_lines()
# print(round(-2.557,1))

# save_predefined_path_coordinates(pts2)
# print(prepend_id_to_path_coord(pts2))
# prepend_id_to_path_coord()
mylist = ['1', '3', '1', '2', '4', '5', '1', '2']
path = "Path 1234"
#
mylist = [*{*mylist}]
# mylist = ['gf', 'rt', 'tyiu', 'bcv', 'rt', 'gs', 'w', 'n3']
ans = sorted(mylist)
# print(ans)
guests = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer']
guests.sort()

# print(mylist)
print("True") if True else False
fruit = 'Apple'
isApple = False
if fruit == 'Apple': isApple = True


# print(isApple)

# print(type(int(path.split()[-1])))
# get_pil_text_size('Hello world in the current', 12, 'times.ttf')


# def plot_multi_subplots():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#     import tkinter as tk
#
#     # Prepare Data
#     x1 = np.linspace(0.0, 5.0)
#     y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
#     x2 = np.linspace(0.0, 3.0)
#     y2 = np.cos(2 * np.pi * x2) * np.exp(-x1)
#
#     # Figure instance
#     fig = plt.Figure()
#
#     # ax1
#     ax1 = fig.add_subplot(441)
#     ax1.plot(x1, y1)
#     ax1.set_title('line plot')
#     ax1.set_ylabel('Damped oscillation')
#
#     # ax2
#     ax2 = fig.add_subplot(442)
#     ax2.scatter(x1, y1, marker='o')
#     ax2.set_title('Scatter plot')
#
#     # ax3
#     ax3 = fig.add_subplot(443)
#     ax3.plot(x2, y2)
#     ax3.set_ylabel('Damped oscillation')
#     ax3.set_xlabel('time (s)')
#
#     # ax4
#     ax4 = fig.add_subplot(444)
#     ax4.scatter(x2, y2, marker='o')
#     ax4.set_xlabel('time (s)')
#
#     # ax5
#     ax5 = fig.add_subplot(445)
#     ax5.scatter(x2, y2, marker='o')
#     ax5.set_xlabel('time (s)')
#
#
#     # When windows is closed.
#
#     def _destroyWindow():
#         root.quit()
#         root.destroy()
#
#
#     # Tkinter Class
#
#     root = tk.Tk()
#     root.withdraw()
#     root.protocol('WM_DELETE_WINDOW', _destroyWindow)  # When you close the tkinter window.
#
#     # Canvas
#     canvas = FigureCanvasTkAgg(fig, master=root)  # Generate canvas instance, Embedding fig in root
#     canvas.draw()
#     canvas.get_tk_widget().pack()
#     #canvas._tkcanvas.pack()
#
#     # root
#     root.update()
#     root.deiconify()
#     root.mainloop()

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


def write_to_video():
    # cap = cv2.VideoCapture(0)     0 for webcam input
    cap = cv2.VideoCapture("../extracted.mp4")
    # Set Dimensions for output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20, (width, height), True)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 0)
            # write the flipped frame
            out.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('Stream disconnected')
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


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
    file_name_path_coord = "../extra_files/saved_traversed_coordinates.csv"
    file_name_path_ccord_id = "../extra_files/all_traversed_coordinates.csv"
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


save_traversed_path_coordinates([pts, pts2])
# write_to_video()
