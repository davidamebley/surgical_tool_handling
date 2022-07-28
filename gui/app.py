"""
This file contains the GUI elements of our app

"""
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import asksaveasfile, asksaveasfilename
from tkinter import scrolledtext
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import cv2
import os
import sys
import datetime
from typing import Iterable
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import asyncio
import asyncio

# Custom variables
username = ""
selected_task = 0
repeat_task_value = 1
performance_results_columns = ['username', 'duration', 'distance_traversed', 'deviation', 'path_id', 'pixel_cm_ratio']
performance_metric_list = ['Duration', 'Distance', 'Deviation']
performance_metric_units = {'Duration': 'secs', 'Distance': 'cm', 'Deviation': 'cm'}
predefined_path_columns = ['path_id', 'length']
predefined_path_coordinates_columns = ['path_id', 'coord_x', 'coord_y']
temp_results = []
predefined_training_paths = []
user_predefined_training_paths = []  # Training paths used by a specific user
selected_predefined_training_path = 1  # Setting default val
predefined_training_path_coordinates = []
returning_user = 0  # Existing user returning to system
is_drawn_new_path = False
results_window = 0
is_view_results_window_open = False
is_view_task_info_window_open = False
current_open_results_window_list = []
current_open_task_info_window_list = []

# ROOT = 'I:\\tool_detection\\david_project\\'
ROOT = 'C:\\Users\\Kwasi\\OneDrive - University of Eastern Finland\\Desktop\\Developer\\CV project\\tool_tracking_deliverable\\'

# Read values from files
# Task types
available_tasks = []
with open('./extra_files/task_type.txt') as inFile:
    available_tasks = [line for line in inFile]

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# Retrieve user list from file
existing_users = []

# Retrieve user performance results from file
performance_results = []


# performance_data = pd.read_csv("./extra_files/results.csv",
#                                names=performance_results_columns)


# ***************************************************************************************************

# define a function for 1st toplevel
# which is associated with root window.
def open_select_task_window():
    # Create widget
    task_window = tk.Toplevel(root)

    # Define title for window
    task_window.title("Select Task")

    # specify size
    task_window.geometry("640x480")

    # Create label
    label_main = tk.Label(task_window, text="Hi, " + username + "! \nPlease select the task type you want to complete",
                          font=("Century", 14))
    label_sub = tk.Label(task_window,
                         text="Click the 'Info' link below to read more about your selected task")
    label_info = tk.Label(task_window, text="Info", cursor="hand2", fg="blue", font=("Century", 10, "underline"))

    # Combobox for Task Type
    combo_task_type = ttk.Combobox(task_window, state="readonly")
    combo_task_type['values'] = tuple(available_tasks)
    combo_task_type.current(0)
    # Bind the multiple events
    for cb in [combo_task_type]:
        cb.bind("<<ComboboxSelected>>", lambda e: set_selected_task_index(combo_task_type.current()))
        cb.bind("<Return>", lambda e: set_selected_task_index(combo_task_type.current()))

    # Custom functions for continue button
    def button_funcs(*funcs):
        def button_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return button_func

    # Create Exit button
    button_cancel = tk.Button(task_window, text="Cancel", bg="gray",
                              command=restore_root_window(task_window.destroy, enable_root_widgets))

    # Button to Continue to Next stage
    button_continue = tk.Button(task_window, text="Continue", fg="white", bg="green",
                                command=button_funcs(task_window.destroy, open_select_predefined_path_window))

    label_main.pack(padx=5, pady=10, side=tk.TOP)
    label_sub.pack(padx=5, pady=10, side=tk.TOP)
    combo_task_type.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    label_info.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    label_info.bind("<Button-1>",
                    lambda e: attempt_open_task_info(combo_task_type.current()))  # Click event
    button_cancel.pack(pady=10, side=tk.BOTTOM)
    button_continue.pack(pady=10, side=tk.BOTTOM)

    # Function to display Task Info
    def attempt_open_task_info(n_current):
        try:
            # users_window.destroy()
            display_task_info(n_current)  # Open Window
        except OSError:
            raise FileNotFoundError("Error opening window")

    # Handle window close event
    task_window.protocol("WM_DELETE_WINDOW", on_closing)

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    task_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    task_window.wm_transient(root)

    # Display until closed manually
    task_window.mainloop()


def open_task_repeat_window():
    # Create widget
    task_repeat_window = tk.Toplevel()

    # define title for window
    task_repeat_window.title("Repeat Task")

    # specify size
    task_repeat_window.geometry("640x480")

    # Create label
    label_main = tk.Label(task_repeat_window, text="Your selected task:\n", font=("Century", 14))
    label_selected_task = tk.Label(task_repeat_window, bg="green", fg="white", text=get_selected_task(),
                                   font=("Century", 12))
    label_sub = tk.Label(task_repeat_window, text="Select the number of times you want to repeat this task:",
                         font=("Century", 13))

    # Spinbox for repeat value
    default_value = tk.StringVar(value=1)
    spinbox_repeat = ttk.Spinbox(task_repeat_window, state="readonly", from_=1, to=10, textvariable=default_value,
                                 wrap=True)

    def set_task_repeat_val():
        global repeat_task_value
        repeat_task_value = int(spinbox_repeat.get())

    # Custom functions for continue button
    def button_funcs(*funcs):
        def button_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return button_func

    # Custom function to display some info when Continue button clicked
    def show_next_window_instruction():
        show_info_messagebox(
            "The next window shows a preview of the path to track. \nPress Spacebar to proceed to the task. Press the 'S' key if you wish to save the path before beginning the task. \nTo quit the started, press the 'Q' key")

    # Create exit button.
    button_cancel = tk.Button(task_repeat_window, text="Cancel", fg="white", bg="gray",
                              command=button_funcs(task_repeat_window.destroy, open_select_predefined_path_window))
    # Continue button
    button_continue = tk.Button(task_repeat_window, text="Continue", fg="white", bg="green",
                                command=button_funcs(set_task_repeat_val, show_next_window_instruction, root.destroy))

    label_main.pack()
    label_selected_task.pack()
    label_sub.pack()
    spinbox_repeat.pack()
    button_cancel.pack(pady=10, side=tk.BOTTOM)
    button_continue.pack(pady=10, side=tk.BOTTOM)

    # Handle window close event
    task_repeat_window.protocol("WM_DELETE_WINDOW", on_closing)

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    task_repeat_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    task_repeat_window.wm_transient(root)

    # Display until closed manually.
    task_repeat_window.mainloop()


def open_select_predefined_path_window():
    # Create widget
    select_path_window = tk.Toplevel(root)

    # Define title for window
    select_path_window.title("Select Path to Train")

    # specify size
    select_path_window.geometry("640x480")

    # Create label
    label_main = tk.Label(select_path_window, text="Choose a 'Path' to train with \n", font=("Century", 14))
    label_sub = tk.Label(select_path_window, text="Select from the list of predefined paths below: \n",
                         font=("Century", 12))
    label_preview_selected_path = tk.Label(select_path_window, text="Preview selected path\n", cursor="hand2",
                                           fg="red",
                                           font=("Century", 10))
    label_draw_new_path = tk.Label(select_path_window, text="Or proceed to draw a new path", cursor="hand2", fg="blue",
                                   font=("Century", 12, "underline"))

    # Combobox for Path list
    combo_path_list = ttk.Combobox(select_path_window, state="readonly")
    temp_list = tuple(get_path_list())
    combo_path_list['values'] = ['Path ' + str(item) for item in temp_list]  # Attach 'Path' string to value
    # combo_path_list['values'] = ["Path " + str(item) for item in temp_list]  # Attach 'Path' string to value
    combo_path_list.current(0)

    # Bind the multiple events
    for cb in [combo_path_list]:
        # cb.bind("<<ComboboxSelected>>", lambda e: print(combo_path_list.current()))
        cb.bind("<<ComboboxSelected>>", lambda e: set_selected_path_id(combo_path_list.current()))
        cb.bind("<Return>", lambda e: set_selected_path_id(combo_path_list.current()))

    # Custom functions for continue button
    def button_funcs(*funcs):
        def button_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return button_func

    # Create Cancel button
    button_cancel = tk.Button(select_path_window, text="Cancel", bg="gray",
                              command=restore_root_window(select_path_window.destroy, enable_root_widgets))

    # Button to Continue to Next stage
    button_continue = tk.Button(select_path_window, text="Continue with selected path", fg="white", bg="green",
                                command=button_funcs(get_path_coordinates, select_path_window.destroy,
                                                     start_proceed_with_existing_path))

    label_main.pack(padx=5, pady=10, side=tk.TOP)
    label_sub.pack(padx=5, pady=10, side=tk.TOP)
    combo_path_list.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    label_preview_selected_path.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    label_preview_selected_path.bind("<Button-1>", lambda e: preview_selected_predefined_path())  # Click event
    label_draw_new_path.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    label_draw_new_path.bind("<Button-1>",
                             lambda e: button_funcs(select_path_window.destroy(), start_draw_new_path()))  # Click event
    button_cancel.pack(pady=10, side=tk.BOTTOM)
    button_continue.pack(pady=10, side=tk.BOTTOM)

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    select_path_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    select_path_window.wm_transient(root)

    # Display until closed manually
    select_path_window.mainloop()


# Function to proceed to draw new path for training
def start_draw_new_path():
    global is_drawn_new_path
    open_task_repeat_window()
    is_drawn_new_path = True


# Function to proceed with pre-existing path for training
def start_proceed_with_existing_path():
    global is_drawn_new_path
    open_task_repeat_window()
    is_drawn_new_path = False


# Function to open existing user list window
def open_existing_users_window():
    # Create widget
    users_window = tk.Toplevel(root)

    # Define title for window
    users_window.title("Existing users")

    # specify size
    users_window.geometry("640x480")

    # Create label
    label_main = tk.Label(users_window, text="Existing users", font=("Century", 14))
    label_sub = tk.Label(users_window, text="Select a name from the list below and proceed with the next action",
                         font=("Century", 12))
    label_view_results = tk.Label(users_window, text="View training results", cursor="hand2", fg="blue",
                                  font=("Century", 12, "underline"))
    label_view_coordinates_traversed = tk.Label(users_window,
                                                text="View Coordinates Traversed", cursor="hand2", fg="blue",
                                                font=("Century", 12, "underline"))

    # Combobox for User list
    combo_user_list = ttk.Combobox(users_window, state="readonly")
    combo_user_list['values'] = tuple(get_user_list())
    combo_user_list.current(0)

    # Bind the multiple events
    for cb in [combo_user_list]:
        cb.bind("<<ComboboxSelected>>", lambda e: set_selected_user_index(combo_user_list.current()))
        cb.bind("<Return>", lambda e: set_selected_user_index(combo_user_list.current()))

    # Custom functions for continue button
    def button_funcs(*funcs):
        def button_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return button_func

    # Custom function to set main session username to returning user
    def set_session_username():
        global username
        username = get_selected_returning_user(combo_user_list.current())

    # Create Cancel button
    button_cancel = tk.Button(users_window, text="Cancel", bg="gray",
                              command=restore_root_window(users_window.destroy, enable_root_widgets))

    # Button to Continue to Next stage
    button_continue = tk.Button(users_window, text="Continue to training", fg="white", bg="green",
                                command=button_funcs(set_session_username, users_window.destroy,
                                                     open_select_task_window))

    label_main.pack(padx=5, pady=10, side=tk.TOP)
    label_sub.pack(padx=5, pady=10, side=tk.TOP)
    combo_user_list.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    label_view_results.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    label_view_results.bind("<Button-1>",
                            lambda e: attempt_open_performance_metrics_window(combo_user_list.current()))  # Click event
    label_view_coordinates_traversed.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    label_view_coordinates_traversed.bind("<Button-1>",
                                          lambda e: attempt_open_coordinates_traversed_window(
                                              combo_user_list.current()))  # Click event
    button_cancel.pack(pady=10, side=tk.BOTTOM)
    button_continue.pack(pady=10, side=tk.BOTTOM)

    # Handle window close event
    # users_window.protocol("WM_DELETE_WINDOW", on_closing)

    # Function to test availability of data before open Performance Metrics window
    def attempt_open_performance_metrics_window(n_current):
        if len(get_user_trained_path_list(get_selected_returning_user(n_current))) > 0:

            set_session_username()  # set main session username to returning user

            users_window.destroy()
            open_performance_metrics_window(n_current)
        else:
            show_error_messagebox("No performance result data found for selected user")

    # Function to test availability of data before open Coordinates traversed window
    def attempt_open_coordinates_traversed_window(n_current):
        # if len(get_user_trained_path_list(get_selected_returning_user(n_current))) > 0:
        if True:

            set_session_username()  # set main session username to returning user

            users_window.destroy()
            open_path_coordinates_traversed_window(n_current)
        else:
            show_error_messagebox("No performance result data found for selected user")

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    users_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    users_window.wm_transient(root)

    # Display until closed manually
    users_window.mainloop()


# Function to open individual performance results
def open_performance_metrics_window(user_id):
    global returning_user
    # Create widget
    performance_metrics_window = tk.Toplevel(root)

    # Define title for window
    performance_metrics_window.title("Performance Results")

    # specify size
    performance_metrics_window.geometry("648x648")

    print("users = ", existing_users)

    # Update Returning User ID
    returning_user = user_id
    # print(f"ID of user is {returning_user}")

    # Create label
    user = get_selected_returning_user(user_id)
    label_main = tk.Label(performance_metrics_window, text="Hi, " + str(user), font=("Century", 14))
    label_sub = tk.Label(performance_metrics_window,
                         text="Select a performance metric below to view",
                         font=("Century", 12))
    # label_sub2 = tk.Label(performance_metrics_window, text="Path", font=("Century", 12))

    # Combobox for Path list
    combo_path_list = ttk.Combobox(performance_metrics_window, state="readonly")
    path_list = tuple(get_user_trained_path_list(user))
    combo_path_list['values'] = ['Path ' + str(item) for item in path_list]
    combo_path_list.current(0)

    # Bind the multiple events
    for cb in [combo_path_list]:
        cb.bind("<<ComboboxSelected>>", lambda e: set_selected_user_index(combo_path_list.current()))
        cb.bind("<Return>", lambda e: set_selected_user_index(combo_path_list.current()))

    # Custom functions for continue button
    def button_funcs(*funcs):
        def button_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return button_func

    # Create Performance metrics frame
    frame_performance_metrics = tk.LabelFrame(performance_metrics_window, text="Performance Metrics", pady=40)

    # Create labels for Frame *TODO NOTE: These will be created dynamically later
    label_duration = tk.Label(frame_performance_metrics, text=performance_metric_list[0], cursor="hand2", fg="blue",
                              font=("Century", 12))
    # label_export_duration = tk.Label(frame_performance_metrics, text="Duration", cursor="hand2", fg="blue",
    #                           font=("Century", 12))
    label_duration.bind("<Button-1>", lambda e: display_results(1, performance_results_columns[1] + f" ({performance_metric_units['Duration']})",
                                                                combo_path_list.get()))  # Click event
    label_distance = tk.Label(frame_performance_metrics, text=performance_metric_list[1], cursor="hand2", fg="blue",
                              font=("Century", 12))
    label_distance.bind("<Button-1>", lambda e: display_results(2, performance_results_columns[2] + f" ({performance_metric_units['Distance']})",
                                                                combo_path_list.get()))  # Click event
    label_deviation = tk.Label(frame_performance_metrics, text=performance_metric_list[2], cursor="hand2", fg="blue",
                               font=("Century", 12))
    label_deviation.bind("<Button-1>", lambda e: display_results(3, performance_results_columns[3] + f" ({performance_metric_units['Deviation']})",
                                                                 combo_path_list.get()))  # Click event

    # Create Export Result frame
    frame_export_result = tk.LabelFrame(performance_metrics_window, text="Export Results", pady=10)

    # Combobox for Export list
    combo_export_list = ttk.Combobox(performance_metrics_window, state="readonly")
    combo_export_list['values'] = [item for item in performance_metric_list]
    combo_export_list.current(0)

    # Bind the multiple events
    # for cb in [combo_export_list]:
    #     cb.bind("<<ComboboxSelected>>", lambda e: set_selected_user_index(combo_export_list.current()))
    #     cb.bind("<Return>", lambda e: set_selected_user_index(combo_export_list.current()))

    # Function to initiate Export result process
    def dispatch_export_result():
        print("Hi, export")
        new_result = 0
        selected_metric = str(combo_export_list.get())
        if selected_metric.casefold() == "duration".casefold():  # caseless comparison
            new_result = get_performance_results(1, combo_path_list.get()), f"duration ({performance_metric_units['Duration']})"  # index for which metric, then selected path
        elif selected_metric.casefold() == "distance".casefold():
            new_result = get_performance_results(2, combo_path_list.get()), f"distance ({performance_metric_units['Distance']})"
        elif selected_metric.casefold() == "deviation".casefold():
            new_result = get_performance_results(3, combo_path_list.get()), f"deviation ({performance_metric_units['Deviation']})"
        else:
            new_result = get_performance_results(1, combo_path_list.get()), f"duration ({performance_metric_units['Duration']})"
        print("Export ", new_result)
        return new_result

    # Function to prepare results for display
    def get_performance_results(result_column_index, selected_path):
        global performance_results
        # print(f"Inside display_results function. ID of user is {returning_user}")
        new_list = []
        performance_results = get_performance_data(result_column_index,
                                                   selected_path)  # index stands for which results column; 1=duration, 2=distance
        print(f"performance_results {performance_results}")
        for i, n_list in enumerate(performance_results):
            print("i, list, performance_results[i]")
            print(i, n_list, performance_results[i])
            new_list.append(float(performance_results[i][0]))
        return new_list

    # Function to display various results based on index passed
    def display_results(index, label, selected_path):
        global performance_results
        new_list = get_performance_results(index, selected_path)
        view_results_window(new_list, label, selected_path)

    # Function to initiate eport for subsequent storage
    def initiate_export_result():
        raw_export_data = dispatch_export_result()
        label = raw_export_data[1]
        export_data = raw_export_data[0]
        columns = [label]
        # flatten_results = flatten_some_list(temp_results)
        dt_frame = pd.DataFrame(export_data)
        dt_frame.index += 1
        # -----------------------
        curr_user = get_username()
        curr_task = get_selected_task()
        curr_path = combo_path_list.get()

        file_types = [('All files','*.*'), ('Comma-separated value files(.csv)', '*.csv')]
        file = asksaveasfilename(initialfile=f'{curr_user}_{curr_task}_{curr_path}_{label}', defaultextension='.csv', filetypes=file_types)
        if file is None or file == '':    # Handle error that appears when cancel without save
            return
        # save ------------------------------ save
        try:
            with open(file, 'w', newline='') as f:
                print("New file created")
                dt_frame.to_csv(f, header=columns, index=True, mode='w', index_label='training_task')
        except FileNotFoundError:
            raise ('Error writing to file')
        show_info_messagebox("File saved successfully in the last directory selected")


    # Close View_Results_window
    def prepare_to_close_view_results():
        global is_view_results_window_open
        if is_view_results_window_open:
            [window.destroy() for window in current_open_results_window_list]
            is_view_results_window_open = False

    # Create Export Button
    button_export = tk.Button(performance_metrics_window, text="Export Result", fg="white", bg="gray",
                              command=initiate_export_result)

    # Create Cancel button
    button_cancel = tk.Button(performance_metrics_window, text="Cancel", bg="gray",
                              command=button_funcs(performance_metrics_window.destroy,
                                                   open_existing_users_window))

    # Button to Exit Program
    button_exit = tk.Button(performance_metrics_window, text="Exit Program", fg="white", bg="red", command=on_closing)

    label_main.pack(padx=5, pady=10, side=tk.TOP)
    label_sub.pack(padx=5, pady=10, side=tk.TOP)
    # label_sub2.pack(padx=5, pady=10, side=tk.TOP)
    combo_path_list.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    # ****** METRICS objects ******
    label_duration.pack(padx=100, pady=10, side=tk.TOP)
    label_distance.pack(padx=0, pady=10, side=tk.TOP)
    label_deviation.pack(padx=0, pady=10, side=tk.TOP)
    combo_export_list.pack(in_=frame_export_result, side=tk.LEFT, padx=20)
    button_export.pack(in_=frame_export_result, side=tk.LEFT, padx=20)
    # **************
    frame_performance_metrics.pack(padx=5, pady=10, side=tk.TOP, fill="x", expand=True)
    frame_export_result.pack(in_=frame_performance_metrics, padx=5, pady=10, side=tk.BOTTOM, expand=True)
    button_cancel.pack(pady=10, side=tk.TOP)
    button_exit.pack(pady=10, side=tk.BOTTOM)

    # Handle window close event
    performance_metrics_window.protocol("WM_DELETE_WINDOW", on_closing)

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    performance_metrics_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    performance_metrics_window.wm_transient(root)

    # Display until closed manually
    performance_metrics_window.mainloop()


# Function to open individual path coordinates traversed
def open_path_coordinates_traversed_window(user_id):
    global returning_user
    # Create widget
    path_coordinates_traversed_window = tk.Toplevel(root)

    # Define title for window
    path_coordinates_traversed_window.title("Coordinates Traversed")

    # specify size
    path_coordinates_traversed_window.geometry("640x480")

    print("users = ", existing_users)

    # Update Returning User ID
    returning_user = user_id
    # print(f"ID of user is {returning_user}")

    # Create label
    user = get_selected_returning_user(user_id)
    label_main = tk.Label(path_coordinates_traversed_window, text="Hi, " + str(user), font=("Century", 14))
    label_sub = tk.Label(path_coordinates_traversed_window,
                         text="Select an item below to view",
                         font=("Century", 12))
    # **********************
    label_view_coordinates = tk.Label(path_coordinates_traversed_window, text="View Coordinates", cursor="hand2",
                                      fg="blue",
                                      font=("Century", 12))
    label_view_coordinates.bind("<Button-1>", lambda e: display_results(1, performance_results_columns[1] + " (secs)",
                                                                        combo_path_list.get()))  # Click event
    label_watch_replay = tk.Label(path_coordinates_traversed_window, text="Watch Replay", cursor="hand2", fg="blue",
                                  font=("Century", 12))
    label_watch_replay.bind("<Button-1>", lambda e: display_results(2, performance_results_columns[2] + " (mm)",
                                                                    combo_path_list.get()))  # Click event

    # Combobox for Path list
    combo_path_list = ttk.Combobox(path_coordinates_traversed_window, state="readonly")
    temp_list = tuple(get_user_trained_path_list(user))
    combo_path_list['values'] = ['Path ' + str(item) for item in temp_list]
    combo_path_list.current(0)

    # Bind the multiple events
    for cb in [combo_path_list]:
        cb.bind("<<ComboboxSelected>>", lambda e: set_selected_user_index(combo_path_list.current()))
        cb.bind("<Return>", lambda e: set_selected_user_index(combo_path_list.current()))

    # Custom functions for continue button
    def button_funcs(*funcs):
        def button_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return button_func

    # Function to display various results based on index passed
    def display_results(index, label, selected_path):
        global performance_results
        # print(f"Inside display_results function. ID of user is {returning_user}")
        new_list = []
        performance_results = get_performance_data(index, selected_path)  # index stands for which results column
        print(f"performance_results {performance_results}")
        for i, list in enumerate(performance_results):
            print("i, list, performance_results[i]")
            print(i, list, performance_results[i])
            new_list.append(float(performance_results[i][0]))
        view_results_window(new_list, label, selected_path)

    # Close View_Results_window
    def prepare_to_close_view_results():
        global is_view_results_window_open
        if is_view_results_window_open:
            [window.destroy() for window in current_open_results_window_list]
            is_view_results_window_open = False

    # Create Cancel button
    button_cancel = tk.Button(path_coordinates_traversed_window, text="Cancel", bg="gray",
                              command=button_funcs(path_coordinates_traversed_window.destroy,
                                                   open_existing_users_window))

    # Button to Exit Program
    button_exit = tk.Button(path_coordinates_traversed_window, text="Exit Program", fg="white", bg="red",
                            command=on_closing)

    label_main.pack(padx=5, pady=10, side=tk.TOP)
    label_sub.pack(padx=5, pady=10, side=tk.TOP)
    # label_sub2.pack(padx=5, pady=10, side=tk.TOP)
    combo_path_list.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    # ****** Link Labels ******
    label_view_coordinates.pack(padx=100, pady=10, side=tk.TOP)
    label_watch_replay.pack(padx=0, pady=10, side=tk.TOP)
    # **************
    button_exit.pack(pady=10, side=tk.BOTTOM)
    button_cancel.pack(pady=10, side=tk.BOTTOM)

    # Handle window close event
    path_coordinates_traversed_window.protocol("WM_DELETE_WINDOW", on_closing)

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    path_coordinates_traversed_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    path_coordinates_traversed_window.wm_transient(root)

    # Display until closed manually
    path_coordinates_traversed_window.mainloop()


# Function to open ALL users' performance results
def open_all_users_performance_metrics_window():
    # Create widget
    performance_metrics_window = tk.Toplevel(root)

    # Define title for window
    performance_metrics_window.title("All Users Performance Results")

    # specify size
    performance_metrics_window.geometry("640x480")

    # Get User List
    users = get_user_list()
    print("users = ", users)

    # Create label
    # user = get_selected_returning_user(user_id)
    label_main = tk.Label(performance_metrics_window, text="All Users ", font=("Century", 14))
    label_sub = tk.Label(performance_metrics_window,
                         text="Select a performance metric below to view",
                         font=("Century", 12))
    # label_sub2 = tk.Label(performance_metrics_window, text="Path", font=("Century", 12))

    # Combobox for Path list
    combo_path_list = ttk.Combobox(performance_metrics_window, state="readonly")
    temp_list = tuple(get_path_list())
    combo_path_list['values'] = ['Path ' + str(item) for item in temp_list]  # Attach 'Path' string to value
    # combo_path_list['values'] = ["Path " + str(item) for item in temp_list]  # Attach 'Path' string to value
    combo_path_list.current(0)

    # Bind the multiple events
    for cb in [combo_path_list]:
        # cb.bind("<<ComboboxSelected>>", lambda e: print(combo_path_list.current()))
        cb.bind("<<ComboboxSelected>>", lambda e: set_selected_path_id(combo_path_list.current()))
        cb.bind("<Return>", lambda e: set_selected_path_id(combo_path_list.current()))

    # Custom functions for continue button
    def button_funcs(*funcs):
        def button_func(*args, **kwargs):
            for f in funcs:
                f(*args, **kwargs)

        return button_func

    # Create Performance metrics frame
    frame_performance_metrics = tk.LabelFrame(performance_metrics_window, text="Performance Metrics", pady=40)

    # Create labels for Frame *TODO NOTE: These will be created dynamically later
    label_duration = tk.Label(frame_performance_metrics, text="Duration", cursor="hand2", fg="blue",
                              font=("Century", 12))
    label_duration.bind("<Button-1>", lambda e: display_results(1, "secs",
                                                                combo_path_list.get(),
                                                                label_duration.cget("text")))  # Click event
    label_distance = tk.Label(frame_performance_metrics, text="Distance", cursor="hand2", fg="blue",
                              font=("Century", 12))
    label_distance.bind("<Button-1>", lambda e: display_results(2, "mm",
                                                                combo_path_list.get(),
                                                                label_duration.cget("text")))  # Click event
    label_deviation = tk.Label(frame_performance_metrics, text="Deviation", cursor="hand2", fg="blue",
                               font=("Century", 12))
    label_deviation.bind("<Button-1>", lambda e: display_results(3, "mm",
                                                                 combo_path_list.get(),
                                                                 label_duration.cget("text")))  # Click event

    # Function to display various results based on index passed
    def display_results(index, label, selected_path, graph_title):
        print("INSIDE THE DISPLAY RESULTS FUNCTION")
        global performance_results
        all_results = []
        converted_results = []
        users_found = []  # users whose data collected
        # print(f"Inside display_results function. ID of user is {returning_user}")
        new_sublist = []
        performance_results, users_with_data = get_performance_data_for_all(index,
                                                                            selected_path)  # index stands for which results column
        for sublist in performance_results:
            print("sublist")
            print(sublist)
            for item in sublist:
                new_sublist.append(float(item))
            # new_list.append(float(performance_results[i][0]))
            converted_results.append(new_sublist)
            new_sublist = []  # empty list
            print(f"converted_results {converted_results}")
        all_results = converted_results
        print(f"all_results {all_results}")
        view_all_results_window(all_results, users_with_data, label, selected_path, graph_title)

    # Close View_Results_window
    def prepare_to_close_view_results():
        global is_view_results_window_open
        if is_view_results_window_open:
            [window.destroy() for window in current_open_results_window_list]
            is_view_results_window_open = False

    # Create Cancel button
    button_cancel = tk.Button(performance_metrics_window, text="Cancel", bg="gray",
                              command=restore_root_window(performance_metrics_window.destroy, enable_root_widgets))

    # Button to Exit Program
    button_exit = tk.Button(performance_metrics_window, text="Exit Program", fg="white", bg="red", command=on_closing)

    label_main.pack(padx=5, pady=10, side=tk.TOP)
    label_sub.pack(padx=5, pady=10, side=tk.TOP)
    # label_sub2.pack(padx=5, pady=10, side=tk.TOP)
    combo_path_list.pack(padx=20, pady=10, side=tk.TOP, fill="x")
    # ****** METRICS Labels ******
    label_duration.pack(padx=100, pady=10, side=tk.TOP)
    label_distance.pack(padx=0, pady=10, side=tk.TOP)
    label_deviation.pack(padx=0, pady=10, side=tk.TOP)
    # **************
    frame_performance_metrics.pack(padx=5, pady=10, side=tk.TOP, fill="x", expand="yes")
    button_cancel.pack(pady=10, side=tk.TOP)
    button_exit.pack(pady=10, side=tk.BOTTOM)

    # Handle window close event
    performance_metrics_window.protocol("WM_DELETE_WINDOW", on_closing)

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    performance_metrics_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    performance_metrics_window.wm_transient(root)

    # Display until closed manually
    performance_metrics_window.mainloop()


# Function to view actual results
def view_results_window(result_list, label, selected_path):
    # Create widget
    global results_window, is_view_results_window_open, current_open_results_window_list
    is_view_results_window_open = True
    results_window = tk.Toplevel(root)

    # Define title for window
    results_window.title("Results")

    # specify size
    results_window.geometry("600x500")

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    results_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    results_window.wm_transient(root)

    # ******************** Graph Stuff 1 *************************
    # the figure that will contain the plot
    fig = Figure(figsize=(5, 5), dpi=100)

    # list of results
    x = [x + 1 for x in range(len(result_list))]
    print(f"x{x}")
    y = result_list

    # adding the subplot
    graph_fig = fig.add_subplot(111)

    # plotting the graph
    graph_fig.plot(x, y, marker="o")
    # plot1.legend()
    graph_fig.set_xlabel("Training task")
    graph_fig.set_ylabel(label)

    # df2.plot(kind='line', legend=True, ax=ax2, color='b', marker='o', fontsize=10)
    graph_fig.set_title(f"User: {username} \n{label} for {selected_path}")

    print(result_list)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    line_graph = FigureCanvasTkAgg(fig, master=results_window)
    line_graph.draw()

    # placing the canvas on the Tkinter window
    line_graph.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(line_graph, results_window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    line_graph.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH)

    # Add this open window to a list
    current_open_results_window_list.append(results_window)

    # Display until closed manually
    results_window.mainloop()


# Function to view ALL users' actual results
def view_all_results_window(result_list, users_found, label, selected_path, graph_title):
    # Create widget
    global results_window, is_view_results_window_open, current_open_results_window_list
    is_view_results_window_open = True
    results_window = tk.Toplevel(root)

    # Define title for window
    results_window.title(f"Results for {selected_path}")

    # specify size
    results_window.geometry("1024x768")

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    results_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    results_window.wm_transient(root)

    # the figure that will contain the plot
    fig = Figure(figsize=(9, 7), dpi=100)

    graph_ax = fig
    # ******************** Graph Stuff *************************
    for i, user_results in enumerate(result_list):
        x, y = 0, 0
        # list of results
        x = [x + 1 for x in range(len(user_results))]
        y = user_results
        print(f"x {x}")
        print(f"y {y}")

        # adding the subplot
        graph_ax = fig.add_subplot(4, 4, i + 1)
        fig.tight_layout()  # Opening up margins around each subplot

        # plotting the graph
        graph_ax.plot(x, y, marker="o")
        # plot1.legend()
        graph_ax.set_xlabel("No. of times trained")
        graph_ax.set_ylabel(label)

        # df2.plot(kind='line', legend=True, ax=ax2, color='b', marker='o', fontsize=10)
        graph_ax.set_title(f"User: [{users_found[i]}] \n{graph_title}")

    # Dynamic Plot Loop Done
    print(result_list)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    line_graph = FigureCanvasTkAgg(fig, master=results_window)
    line_graph.draw()

    # placing the canvas on the Tkinter window
    line_graph.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(line_graph, results_window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    line_graph.get_tk_widget().pack(side=tk.LEFT)

    # Add this open window to a list
    current_open_results_window_list.append(results_window)

    # Display until closed manually
    results_window.mainloop()


# Close View Task Info Window
def prepare_to_close_view_task_info():
    global is_view_task_info_window_open
    if is_view_task_info_window_open:
        [window.destroy() for window in current_open_task_info_window_list]
        is_view_task_info_window_open = False


# Function to read task info
def view_task_info_window(task_name):
    tasks_root_dir = '../extra_files/task_info/'
    file_path = ''
    # Check which task selected
    if task_name == 'Path tracking':
        file_path = 'path_tracking.txt'
    #     Add more logic later based on task additions

    # Create widget
    global is_view_task_info_window_open, current_open_task_info_window_list
    is_view_task_info_window_open = True
    task_info_window = tk.Toplevel(root)

    # Define title for window
    task_info_window.title("Task Info")

    # specify size
    task_info_window.geometry("600x500")

    # Position window relative to root
    win_x = root.winfo_x()
    win_y = root.winfo_y()
    task_info_window.geometry("+%d+%d" % (win_x + 200, win_y + 200))

    # Place window at the top
    task_info_window.wm_transient(root)

    # ******************** Data Stuff *************************
    text_content = 'Sorry, nothing to display for now'
    try:
        with open(tasks_root_dir + file_path) as f:
            text_content = f.read()
            print(text_content)
    except OSError:
        raise FileNotFoundError("Error reading file")

    # Add a label To display Text
    label = tk.Label(task_info_window, text=text_content, font=('Aerial', 17))
    label.pack()

    # Add this open window to a list
    current_open_task_info_window_list.append(task_info_window)

    # Display until closed manually
    task_info_window.mainloop()


# Function to view Selected Predefined Training Path
def preview_selected_predefined_path():
    pts = get_path_coordinates()
    img_poly = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
    img2 = img_poly.copy()
    cv2.namedWindow('Preview')
    if len(pts) > 1:

        for i in range(len(pts) - 1):
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 255, 255), thickness=2)

    cv2.imshow('Preview', img2)
    cv2.waitKey()


# Function to display more info of selected task. THIS IS AN ALTERNATIVE TO FUNC (View Task Info WIndow)
def display_task_info(task_name):
    tasks_root_dir = 'extra_files\\task_info\\'
    file_path = ''
    # Check which task selected
    if task_name == 'Path tracking':
        file_path = 'path_tracking.txt'
    #     Add more logic later based on task additions

    # Create widget
    global is_view_task_info_window_open, current_open_task_info_window_list
    is_view_task_info_window_open = True
    # Create widget
    info_window = tk.Toplevel()

    # define title for window
    info_window.title("Task Info")

    # specify size
    info_window.geometry("480x320")

    # ******************** Data Stuff *************************
    text_content = 'Sorry, nothing to display for now'
    try:
        assert os.path.isfile(ROOT + 'extra_files\\task_info\\path_tracking.txt')
        with open(ROOT + 'extra_files\\task_info\\path_tracking.txt', "r") as f:
            text_content = f.read()
            print(text_content)
    except OSError:
        raise FileNotFoundError("Error reading file")

    # Create label
    label_main = tk.Label(info_window, text=get_selected_task())

    # Where to display text
    label_text = scrolledtext.ScrolledText(info_window, bg='white', fg='blue', relief=tk.GROOVE, height=600,
                                           font='TkFixedFont', wrap='word')
    label_text.insert(tk.END, text_content)

    # Okay button
    button_okay = tk.Button(info_window, text="Continue", command=info_window.destroy)

    label_main.pack()
    label_text.pack()
    button_okay.pack()

    # Add this open window to a list
    current_open_task_info_window_list.append(info_window)

    # Display until closed manually.
    info_window.mainloop()


# Function to convert performance index to actual names
def convert_performance_index(index):
    for ind in range(len(performance_results_columns)):
        if index == ind:
            return performance_results_columns[index]


# Function to retrieve performance data
def get_performance_data(performance_index, selected_path):
    results = []
    n_path = selected_path.split()[-1]  # Extract the number part of the string
    performance_metric = convert_performance_index(performance_index)  # get the str equivalence of metric
    performance_data = get_performance_results_csv()
    pixel_cm_ratio = 0
    # print(f"Inside get_performance_data function. ID of user is {returning_user}")
    print(f"Inside get_performance_data function. Path ID is {n_path}")
    for row in performance_data.itertuples(index=False):
        if row.username == get_selected_returning_user(returning_user):  # get results based on which user
            if row.path_id == n_path:  # get values based on path id
                # is_user_present = True
                pixel_cm_ratio = row.pixel_cm_ratio
                result = 0
                if performance_index == 2 or performance_index == 3:
                    result = [float(getattr(row, performance_metric)) / float(
                        pixel_cm_ratio)]  # do pixel-cm-ratio conv. here for graph
                else:
                    result = [getattr(row, performance_metric)]  # return raw results in time/secs value
                results.append(result)
    return results


# Function to retrieve performance data
def get_performance_data_for_all(performance_index, selected_path):
    result = []
    results = []
    result_list = []
    # n_users = []
    user_list = []
    is_current_user_result_started_being_recorded = False
    is_user_path_match = False
    # Get User List
    users = get_user_list()
    n_path = selected_path.split()[-1]  # Extract the number part of the string
    performance_metric = convert_performance_index(performance_index)
    performance_data = get_performance_results_csv()
    # print(f"Inside get_performance_data function. ID of user is {returning_user}")
    print(f"Inside get_performance_data function. Path ID is {n_path}")
    for user in users:
        if user not in user_list:
            for row in performance_data.itertuples(index=False):  # Loop over all results for current user results
                if row.username == user:
                    if row.path_id == n_path:  # if user found in result list and path match
                        # is_user_present = True
                        is_user_path_match = True
                        if not is_current_user_result_started_being_recorded:
                            is_current_user_result_started_being_recorded = True
                            user_list.append(user)  # add user to list of those with this performance result
                        result = getattr(row, performance_metric)
                        results.append(result)
            #
            if is_user_path_match:
                result_list.append(results)
                print(f"result_list.append(results) ====> {results}")
                results = []  # empty the list
                is_user_path_match = False
        #
        is_current_user_result_started_being_recorded = False
    print(f"n_users===> {user_list}")
    print(f"return result_list ====> {result_list}")
    return result_list, user_list


# Function to get user list
def get_user_list():
    global existing_users
    user_list = []
    user_data = get_user_csv()
    for row in user_data.itertuples(index=False):
        result = row.username
        user_list.append(result)
    existing_users = user_list
    return user_list


# Function to get trained Path list associated with user's results
def get_user_trained_path_list(n_user):
    global user_predefined_training_paths
    user_path_list = []
    results_data = get_performance_results_csv()
    for row in results_data.itertuples(index=False):
        if row.username == n_user:
            result = row.path_id
            user_path_list.append(result)
    user_path_list = [*{*user_path_list}]  # Remove duplicates
    sorted_list = sorted(user_path_list)
    user_predefined_training_paths = sorted_list
    return user_predefined_training_paths


# Function to Get Selected Task String
def get_selected_task():
    task = available_tasks[selected_task]  # Get task with index
    return task


# Function to get predefined training path(id) from list
def get_path_list():
    global predefined_training_paths
    path_list = []
    path_data = get_path_csv()
    for row in path_data.itertuples(index=False):
        result = row.path_id
        path_list.append(result)
    predefined_training_paths = path_list
    return path_list


# Function to get selected predefined training path coordinates
def get_path_coordinates():
    global predefined_training_path_coordinates
    selected_id = selected_predefined_training_path
    print("ID of selected path is ", selected_id)
    path_coords = []
    path_data = get_path_coordinates_csv()
    for row in path_data.itertuples(index=False):
        if row.path_id == selected_id:
            result = tuple((row.coord_x, row.coord_y))
            path_coords.append(result)
    predefined_training_path_coordinates = path_coords
    # print(predefined_training_path_coordinates)
    return path_coords


# Function to Retrieve user performance results from csv file
def get_performance_results_csv():
    try:
        performance_data = pd.read_csv("./extra_files/results.csv", names=performance_results_columns)
        return performance_data
    except OSError:
        raise FileNotFoundError("Error locating Results csv file")


# Function to retrieve path details from csv file
def get_path_csv():
    # Retrieve path from file
    try:
        path_data = pd.read_csv("./extra_files/training_paths.csv", names=predefined_path_columns, header=0)
        return path_data
    except OSError:
        raise FileNotFoundError("Error locating training path file")


# Function to retrieve user data from csv file
def get_user_csv():
    # Retrieve user list from file
    try:
        user_data = pd.read_csv("./extra_files/users.csv", names=['username'], header=0)
        return user_data
    except OSError:
        raise FileNotFoundError("Error locating users file")


# Function to retrieve path coordinates from csv file
def get_path_coordinates_csv():
    # Retrieve path from file
    try:
        path_data = pd.read_csv("./extra_files/saved_path_coordinates.csv", names=predefined_path_coordinates_columns,
                                header=0)
        return path_data
    except FileNotFoundError:
        print("Error locating training path file")


# Function to Get Selected Returning User from list
def get_selected_returning_user(user_id=returning_user):
    user = existing_users[user_id]  # Get user with index
    # user = existing_users[returning_user]  # Get user with index
    return user


# Function to Set Selected Path ID from list
def set_selected_path_id(index):
    global selected_predefined_training_path
    selected_path_id = predefined_training_paths[index]  # Get path with index
    selected_predefined_training_path = selected_path_id


# Function to Set the index value of 'Selected task'
def set_selected_task_index(index):
    global selected_task
    selected_task = index


# Function to Set the index value of Returning User
def set_selected_user_index(index):
    global returning_user
    returning_user = index


def get_username():
    return username


def save_username(txt_user):
    global username
    username = txt_user
    if is_username_exists(username):
        show_error_messagebox("Username already exists. Please try a new one")
    else:
        # Export data to external file
        file_name = './extra_files/users.csv'
        columns = ['username']
        dt_frame = pd.DataFrame([username])
        # dt_frame.to_csv(file_name, mode='a', header=not os.path.exists(file_name))
        try:
            if not os.path.isfile(file_name):  # If not exist
                with open(file_name, 'a', newline='') as f:
                    print("New file created")
                    dt_frame.to_csv(f, header=columns, index=False, mode='a')
            else:
                with open(file_name, 'a', newline='') as f:
                    dt_frame.to_csv(f, header=(f.tell() == 0), index=False, mode='a')
            # ---------------- Proceed to Next Window ------------------
            disable_root_widgets()  # disable root widgets
            open_select_task_window()
        except OSError:
            raise FileNotFoundError('Error writing to file')


# Function to check if username exists before registering
def is_username_exists(n_user):
    user_data = get_user_csv()
    # Check for name existence
    for row in user_data.itertuples(index=False):
        if row.username == n_user:
            return True
    return False


# Function to append results to list
def append_results(results):
    global temp_results
    user_name = [get_username()]
    results = user_name + list(results)  # prepend username to results in a new list
    flatten_results = flatten_some_list(results)
    temp_results.append(flatten_results)


# Function to save/export performance results to file
def save_results():
    columns = performance_results_columns
    # flatten_results = flatten_some_list(temp_results)
    dt_frame = pd.DataFrame(temp_results)
    file_name = './extra_files/results.csv'
    try:
        if not os.path.isfile(file_name):
            with open(file_name, 'a', newline='') as f:
                print("New file created")
                dt_frame.to_csv(f, header=columns, index=False, mode='a')
                # print("FINAL", flatten_results)

        else:
            with open(file_name, 'a', newline='') as f:  # if previous file exists
                dt_frame.to_csv(f, header=(f.tell() == 0), index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)
    except FileNotFoundError:
        print('Error writing to file')
    print('Saved results: ', temp_results)


# Function to export individual results to file upon request
def export_individual_results():
    columns = performance_results_columns
    # flatten_results = flatten_some_list(temp_results)
    dt_frame = pd.DataFrame(temp_results)
    file_name = './extra_files/results.csv'
    try:
        if not os.path.isfile(file_name):
            with open(file_name, 'a', newline='') as f:
                print("New file created")
                dt_frame.to_csv(f, header=columns, index=False, mode='a')
                # print("FINAL", flatten_results)

        else:
            with open(file_name, 'a', newline='') as f:
                dt_frame.to_csv(f, header=(f.tell() == 0), index=False, mode='a')
                print("Data saved successfully")
                # print("FINAL", flatten_results)
    except FileNotFoundError:
        print('Error writing to file')
    print('Saved results: ', temp_results)


# Function to record screen
async def write_to_video(src, output_path):
    # cap = cv2.VideoCapture(0)     0 for webcam input
    print("Async video writer called")
    video_source = src  # 0 for webcam, "../path.extension" for vid files
    current_user = username
    task_selected = get_selected_task()  # which training task
    current_datetime = datetime.datetime.now()
    current_dt = current_datetime.strftime("%Y%m%d%H%M%S")  # YearMonthDayHourMinuteSec
    vid_output_path = output_path
    current_year_month = current_datetime.strftime("%Y%m")
    cap = cv2.VideoCapture(video_source)  # "../extracted.mp4"
    cap.set(3, 640)
    cap.set(4, 480)
    # Set Dimensions for output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create YearMonth Folder if not exist
    if not os.path.exists(vid_output_path + '/' + current_year_month):
        os.makedirs(vid_output_path + '/' + current_year_month)
        # create variable for save folder path
        save_dir = vid_output_path + '/' + current_year_month
    else:
        # create variable for save folder path
        save_dir = vid_output_path + '/' + current_year_month

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_dir + '/' + current_user + '_' + task_selected + '_' + current_dt + '.mp4', fourcc, 20,
                          (640, 480), True)
    # out = cv2.VideoWriter('test.mp4', fourcc, 20,
    #                       (640, 480), True)

    while cap.isOpened():
        print("Writing to video...")
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 0)
            # write the flipped frame
            out.write(frame)
            # cv2.imshow('frame', frame)
            print("Writing to /recording video")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('Stream disconnected')
            print(current_datetime.strftime("%Y:%m:%d %H:%M:%S"))
            break

    # while cap.isOpened():
    # ret, frame = cap.read()
    # if ret:
    #     frame = cv2.flip(frame, 0)
    #     # write the flipped frame
    #     out.write(frame)
    #     # cv2.imshow('frame', frame)
    #     print("Writing to /recording video")
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     #     break
    # else:
    #     print('Stream disconnected')
    #     print(current_datetime.strftime("%Y:%m:%d %H:%M:%S"))
    #     # break
    #     cap.release()
    #     out.release()

    # Release everything if job is finished
    cap.release()
    out.release()

    # cv2.destroyAllWindows()


# Function to Calculate Frames Per Sec
def get_frames_per_sec(src):
    # Create video capture object
    video_source = src
    cap = cv2.VideoCapture(video_source)

    if video_source == 0 or video_source == "0":  # if webcam
        # Number of frames to capture. An optimal val for calculation
        num_frames = 120  # Capturing 120 frames

        # Start time
        start = time.time()

        # Grab a few frames
        for i in range(0, num_frames):
            ret, frame = cap.read()

        # End time
        end = time.time()

        # Time elapsed
        seconds = end - start
        # print("Time taken : {0} seconds".format(seconds))
        # Calculate frames per second
        fps = num_frames / seconds
        print("Estimated Webcam Frames Per Second : {0}".format(fps))
    else:
        if int(major_ver) < 3:
            fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


# Flatten some list of list
def flatten_some_list(n_list):  # converting a 2D list into a 1D list
    flat_list = []
    for sublist in n_list:
        if isinstance(sublist, Iterable) and not isinstance(sublist, (str, bytes)):  # Check if item not string
            for item in sublist:
                flat_list.append(item)
        else:
            flat_list.append(sublist)
    return flat_list


# Function to perform Main Start Button Click Commands
def start_button_click(e=None):
    text_username = textbox_username.get()
    text_username = ''.join(text_username.split())
    if text_username == "":
        show_info_messagebox("Please enter a username")
    else:
        save_username(text_username)  # Attempt save username before procession


# Function to show info messagebox
def show_info_messagebox(message):
    messagebox.showinfo("Info", message)


# Function to show warning messagebox
def show_warning_messagebox(message):
    messagebox.showwarning("Warning", message)


# Function to show error messagebox
def show_error_messagebox(message):
    messagebox.showerror("Error", message)


# Function to disable Root widgets
def disable_root_widgets():
    widgets = [button_start, textbox_username, label_existing_user]
    label_existing_user.unbind("<Button-1>")
    # label_view_all_results.unbind("<Button-1>")
    for widget in widgets:
        if widget['state'] == "normal":
            widget["state"] = "disable"


# Function to disable Root widgets
def enable_root_widgets():
    widgets = [button_start, textbox_username, label_existing_user]
    label_existing_user.bind("<Button-1>", lambda e: prepare_to_open_existing_users_window())  # enable click event
    # label_view_all_results.bind("<Button-1>",
    #                             lambda e: prepare_to_open_all_users_performance_metrics_window())  # enable click event
    for widget in widgets:
        if widget['state'] == "disabled":
            widget["state"] = "normal"


# Custom function of functions to restore the main root window from another window
def restore_root_window(*funcs):
    def button_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)

    return button_func


# Handle window close event
def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit the program?"):
        root.destroy()
        sys.exit()


# Our Main Window
root = tk.Tk()
root.title('Surgical Instrument Training')
root.geometry("720x720")

# Two Frames TOP and BOTTOM
top_frame = tk.Frame(root).pack()
bottom_frame = tk.Frame(root).pack(side="bottom")

# Logo Image as label
logo = Image.open('./images/app_images/logo9.png')
logo = ImageTk.PhotoImage(logo)
label_logo = tk.Label(top_frame, image=logo).pack()

# Instruction label text
label_instructions = tk.Label(root, text="Please type your name to begin", font=("Century", 18, "bold"), height=6)
label_instructions.pack()

# Username label text
label_username = tk.Label(bottom_frame, text="Username", font="Century", height=2)
label_username.pack(padx=20, pady=5)

# Username Textbox
textbox_username = tk.Entry(bottom_frame, width=25, font=(None, 14))  # We use 'Entry' for one line text input
textbox_username.bind("<Return>", lambda e: start_button_click())  # Proceed when Return key pressed
textbox_username.focus()  # Set focus on widget
textbox_username.pack(padx=20, pady=5)

# label links
label_existing_user = tk.Label(bottom_frame, text="Already a user? Click here to proceed", cursor="hand2", fg="blue",
                               font=("Century", 14, "underline"))
# label_view_all_results = tk.Label(bottom_frame, text="View results for all users", cursor="hand2", fg="green",
#                                   font=("Century", 11))
label_existing_user.pack(padx=20, pady=5)
# label_view_all_results.pack(padx=20, pady=5)
label_existing_user.bind("<Button-1>", lambda e: prepare_to_open_existing_users_window())  # Click event
# label_view_all_results.bind("<Button-1>",
#                             lambda e: prepare_to_open_all_users_performance_metrics_window())  # Click event


def prepare_to_open_existing_users_window():
    disable_root_widgets()  # disable root widgets
    open_existing_users_window()


def prepare_to_open_all_users_performance_metrics_window():
    disable_root_widgets()  # disable root widgets
    open_all_users_performance_metrics_window()


# Start Button
button_start = tk.Button(bottom_frame, command=start_button_click, text="Start", font="Century",
                         bg="green", fg="white",
                         height=1, width=15)
button_start.pack(padx=5, pady=20, side=tk.BOTTOM)

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()
