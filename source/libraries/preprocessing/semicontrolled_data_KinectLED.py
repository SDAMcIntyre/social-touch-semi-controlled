import csv
import cv2
import numpy as np

from ..misc.time_cost_function import time_it


class KinectLED:
    def __init__(self):
        self.cap = None
        self.circle_center = None
        self.circle_radius = None
        self.drawing = False
        self.confirmation_result = None
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])
        self.first_frame = None
        self.background_mask = None
        self.threshold_value = None

        # video info
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.frame_count = None
        self.fourcc_str = None

        # results
        self.green_levels = []
        self.led_on = []
        self.dt = []

    def initialise(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file.")

        ret, self.first_frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read the first frame.")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        # Convert fourcc to a string
        self.fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.circle_center = (x, y)
            self.circle_radius = 0

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.circle_radius = int(np.sqrt((x - self.circle_center[0])**2 + (y - self.circle_center[1])**2))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.circle_radius = int(np.sqrt((x - self.circle_center[0])**2 + (y - self.circle_center[1])**2))

    def select_led(self):
        if self.first_frame is None:
            raise Exception("Error: First frame not available. Please initialize the video first.")

        while True:
            cv2.namedWindow('First Frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('First Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback('First Frame', self.draw_circle)

            while True:
                frame_copy = self.first_frame.copy()
                if self.circle_center and self.circle_radius:
                    cv2.circle(frame_copy, self.circle_center, self.circle_radius, (0, 255, 0), 2)
                cv2.imshow('First Frame', frame_copy)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or (not self.drawing and self.circle_center and self.circle_radius):
                    break

            cv2.destroyAllWindows()

            if not self.circle_center or not self.circle_radius:
                raise Exception("Error: No circle was drawn.")

            # Display the confirmation window with buttons
            self.confirmation_result = None
            confirmation_window = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(confirmation_window, "Is the circle set correctly?", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            cv2.rectangle(confirmation_window, (50, 100), (150, 150), (0, 255, 0), -1)
            cv2.putText(confirmation_window, "Yes", (70, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(confirmation_window, (250, 100), (350, 150), (0, 0, 255), -1)
            cv2.putText(confirmation_window, "No", (270, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('Confirmation', confirmation_window)

            def confirm_circle(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if 50 < x < 150 and 100 < y < 150:
                        self.confirmation_result = True
                    elif 250 < x < 350 and 100 < y < 150:
                        self.confirmation_result = False
                    cv2.destroyAllWindows()

            cv2.setMouseCallback('Confirmation', confirm_circle)

            while self.confirmation_result is None:
                cv2.waitKey(1)

            if self.confirmation_result:
                break
            else:
                self.circle_center = None
                self.circle_radius = None

        # Create the background mask based on the selected circle
        self.background_mask = np.zeros_like(self.first_frame)
        cv2.circle(self.background_mask, self.circle_center, self.circle_radius, (255, 255, 255), -1)
        self.background_mask = cv2.bitwise_and(self.first_frame, self.background_mask)

    @time_it
    def monitor_green_levels(self, threshold, show=False):
        if self.cap is None or self.first_frame is None:
            raise Exception("Error: Video not initialized. Please call initialise() first.")

        # Reset to the first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if show:
            cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        nframes_predicted = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        nframes = 0
        while True:
            correctly_read, frame = self.cap.read()
            if not correctly_read:
                break
            nframes += 1
            print("Frame:", nframes, "/", nframes_predicted)

            # Subtract the background
            foreground = cv2.absdiff(frame, self.background_mask)

            # Convert the frame to the HSV color space
            hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

            # Create a mask for the green color
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

            # Create a mask for the circular ROI
            circle_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.circle(circle_mask, self.circle_center, self.circle_radius, 255, -1)

            # Combine the green mask and the circle mask
            combined_mask = cv2.bitwise_and(mask, mask, mask=circle_mask)

            # Calculate the average green level in the circular AOI
            AOI = cv2.bitwise_and(frame, frame, mask=combined_mask)
            avg_green = np.mean(AOI[:, :, 1])  # Average of the green channel

            # Store the average green level
            self.green_levels.append(avg_green)

            if show:
                # Draw the circle on the frame (optional)
                cv2.circle(frame, self.circle_center, self.circle_radius, (0, 255, 0), 2)
                # Display the frame (optional)
                cv2.imshow('Frame', frame)

        self.cap.release()
        cv2.destroyAllWindows()

        # define the ON/OFF time series
        self.threshold_value = threshold
        self.led_on = [green > self.threshold_value for green in self.green_levels]

        self.frame_count = nframes
        # Create the vector dt of delta times
        self.dt = np.linspace(0, 1/self.fps * (self.frame_count - 1), self.frame_count)

    def get_green_levels(self):
        return self.green_levels

    def save_to_csv(self, file_path, file_name):
        full_path = f"{file_path}/{file_name}"
        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["dt", "led_on"])
            for dt_value, led_value in zip(self.dt, self.led_on):
                writer.writerow([dt_value, led_value])
