"""Detect marker in background with distinct color with mouse clicks. Right-click on image to get BGR values, left-click for coordinates"""
import numpy as np
import cv2

# load a camera cap
source = "detect_red3.MOV"  # variable to hold input feed, i.e, webcam or vid file. 0 for webcam; str.extension for file
cap_source = source
accumulated_w = 0
accumulated_h = 0
average_ratio = 0
average_ratio_n = 100  # number of elements or times for average calculation
counter = 0  # for average calculation
color_threshold = 4000


def detect_marker_in_background(cap):
    global counter, average_ratio, average_ratio_n
    # Read until video is completed
    cap = cv2.VideoCapture(cap)
    some_test = 0
    while cap.isOpened():
        # SOme random loop count test. Not necessary for program running
        some_test += 1
        print(f"MAIN PROGRAM LOOP... {some_test}")
        # Read frame-by-frame
        ret, image = cap.read()
        # Get height and width of main frame
        frame_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_w = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Define Lower and Upper boundaries of color
        # lower_color = np.array([79, 64, 190], dtype="uint8")
        # upper_color = np.array([89, 72, 203], dtype="uint8")
        lower_color = np.array([73, 71, 131], dtype="uint8")
        upper_color = np.array([74, 66, 144], dtype="uint8")
        print("Marker detect on")

        if ret:
            # Find the colors within the specified boundaries and apply the mask
            # Return a binary mask of white pixels (255) pixels that fall into the boundaries. Black pixels (0) do not fall.
            mask = cv2.inRange(image, lower_color, upper_color)
            print("Mask, ", mask)

            # Highlight the area where the color is detected in the input image with Bitwise AND
            output = cv2.bitwise_and(image, image, mask=mask)

            # ******************************* More Image Fine-tuning *************************************

            # Convert Image to grayscale
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

            # Create a Mask with adaptive threshold
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print("contours", contours)

            object_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > color_threshold:     # if enough color detected. 4000 optimal
                    object_contours.append(cnt)

            # Draw objects boundaries
            global box, w, h
            for cnt in object_contours:
                counter += 1
                # Get rect
                rect = cv2.minAreaRect(cnt)  # MinAreaRect == Where we're going to gen. our rectangle from
                (x, y), (w, h), angle = rect

                # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # Display box size
                print("box... ", box)

                # Pixel_cm_ratio
                pixel_cm_ratio = (h * 2 + w * 2) / 12
                if counter < average_ratio_n:  # get a few vals of calculated ratios
                    average_ratio += pixel_cm_ratio
                    print("Calculating Average")
                else:  # stop calculation
                    print("Average ratio after {} counts is {}".format(counter, average_ratio / counter))
                    # Release video capture object when all done
                    cap.release()
                    cv2.destroyAllWindows()
                    return average_ratio / counter

                print("pixel ratio", pixel_cm_ratio)
                print("Px height is {}, and width is {}".format(h, w))

                # Display width on screen
                cv2.putText(image, "Length {}cm".format(round(h / pixel_cm_ratio, 1)), (int(x), int(y)),
                            cv2.FONT_HERSHEY_PLAIN, 2,
                            (100, 200, 0), 2)
                cv2.putText(image, "Width {}cm".format(round(w / pixel_cm_ratio, 1)), (int(x), int(y + 100)),
                            cv2.FONT_HERSHEY_PLAIN, 2,
                            (100, 200, 0), 2)

                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.polylines(image, [box], True, (255, 0, 0), 2)

            cv2.imshow("Image", image)
            key = cv2.waitKey(1)  # this for video capture to be able to break the video
            if key == 27:
                break

        # Break the loop
        else:
            break

    # Release video capture object when all done
    cap.release()
    cv2.destroyAllWindows()
    return average_ratio

# Function to record screen
def write_to_video(src, output_path):
    # cap = cv2.VideoCapture(0)     0 for webcam input
    video_source = src  # 0 for webcam, "../path.extension" for vid files
    vid_output_path = output_path
    cap = cv2.VideoCapture(video_source)  # "../extracted.mp4"
    cap.set(3, 640)
    cap.set(4, 480)
    # Set Dimensions for output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(save_dir + '/' + current_user + '_' + task_selected + '_' + current_dt + '.mp4', fourcc, 20,
    #                       (width, height), True)
    out = cv2.VideoWriter('test.mp4', fourcc, 20,
                          (640, 480), True)

    while cap.isOpened():
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
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    # cv2.destroyAllWindows()

import asyncio

# async def main():
#     print('Hello ...')
#     # await asyncio.sleep(1)
#     print('... World!')

# asyncio.run(main())

# Run Function
# detect_marker_in_background(cap_source)
# write_to_video(0, '')
