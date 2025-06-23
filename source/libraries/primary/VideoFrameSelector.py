
import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Toplevel
from PIL import Image, ImageTk
import cv2

class VideoFrameSelector:
    """
    A GUI class that allows a user to navigate a video with a slider
    and select a specific frame.
    """
    def __init__(self, master, video_path):
        """
        Initializes the VideoFrameSelector GUI.

        Args:
            master (tk.Tk or tk.Toplevel): The parent window for the GUI.
            video_path (str): The path to the video file.
        """
        self.master = master
        self.video_path = video_path
        self.selected_frame_id = None
        self.selected_frame_image = None

        # Open the video file with OpenCV
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file at {self.video_path}")
            self.master.destroy()
            return

        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Configure the main window
        self.master.title("Video Frame Selector")
        
        # --- Create and pack widgets ---

        # Label to display the video frame
        self.image_label = Label(self.master)
        self.image_label.pack(pady=10, padx=10)

        # Label to display the current frame number
        self.frame_info_label = Label(self.master, text="Frame: 0 / {}".format(self.total_frames - 1))
        self.frame_info_label.pack()

        # Slider for navigating frames
        self.slider = Scale(
            self.master,
            from_=0,
            to=self.total_frames - 1,
            orient="horizontal",
            command=self._update_frame_from_slider,
            length=max(600, self.video_width) # Make slider at least 600px or video width
        )
        self.slider.pack(pady=5, padx=20, fill='x')

        # Button to save the frame ID and close the GUI
        self.save_button = Button(
            self.master,
            text="Save Frame ID",
            command=self._save_and_quit
        )
        self.save_button.pack(pady=10)

        # --- Initial setup ---
        
        # Display the first frame
        self._update_frame(0)
        
        # Set the window closing protocol
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Keyboard Bindings ---
        self.master.bind("<KeyPress>", self._on_key_press)
        # Set focus to the main window to receive key events
        self.master.focus_set()

        """Centers a tkinter window on the screen."""
        self.master.update_idletasks()  # Crucial: calculates the window's size
        # Get the window's calculated width and height
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        # Get the screen's width and height
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        # Calculate the position for the top-left corner
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        # Set the window's geometry
        self.master.geometry(f'{width}x{height}+{x}+{y}')


    def _on_key_press(self, event):
        """Handles keyboard arrow key presses for frame navigation."""
        current_frame = self.slider.get()
        new_frame = current_frame

        # Check for the Control key modifier (state mask 0x4)
        is_ctrl_pressed = (event.state & 0x4) != 0
        step = 10 if is_ctrl_pressed else 1

        # Determine new frame based on arrow key
        if event.keysym == 'Right':
            new_frame = current_frame + step
        elif event.keysym == 'Left':
            new_frame = current_frame - step
        
        # Clamp the new frame value to be within video bounds
        clamped_frame = max(0, min(new_frame, self.total_frames - 1))

        # Update the slider if the frame has changed
        if clamped_frame != current_frame:
            self.slider.set(clamped_frame)

    def _update_frame_from_slider(self, slider_value):
        """Callback for when the slider is moved."""
        self._update_frame(int(slider_value))

    def _update_frame(self, frame_id):
        """
        Sets the video to a specific frame and updates the image label.

        Args:
            frame_id (int): The frame number to display.
        """
        # Set the video capture to the desired frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        ret, frame = self.cap.read()
        if ret:
            # Update the frame info label
            self.frame_info_label.config(text=f"Frame: {frame_id} / {self.total_frames - 1}")

            # Convert the OpenCV frame (BGR) to a PIL Image (RGB)
            cv2_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(cv2_image)

            # --- Resizing for display to prevent overly large windows ---
            max_display_height = 720
            if pil_image.height > max_display_height:
                ratio = max_display_height / pil_image.height
                new_width = int(pil_image.width * ratio)
                # MODIFICATION: Changed Image.Resampling.LANCZOS to Image.LANCZOS for backward compatibility
                pil_image = pil_image.resize((new_width, max_display_height), Image.LANCZOS)
            
            # Convert the PIL Image to a Tkinter PhotoImage
            photo_image = ImageTk.PhotoImage(image=pil_image)
            
            # Update the image label
            self.image_label.config(image=photo_image)
            # IMPORTANT: Keep a reference to the image to prevent it from being garbage collected
            self.image_label.image = photo_image

    def _save_and_quit(self):
        """
        Saves the current frame ID and closes the application.
        """
        self.selected_frame_id = self.slider.get()
        print("Frame ID saved.")

        # Set the video to the selected frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.selected_frame_id)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert the full-resolution frame to an RGB PIL Image and store it
            cv2_image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.selected_frame_image = Image.fromarray(cv2_image_rgb)
            print(f"Frame {self.selected_frame_id} stored as an RGB image.")
        else:
            print(f"Failed to read frame {self.selected_frame_id} for saving.")

        self._on_close()

    def _on_close(self):
        """
        Handles the window closing event. Releases resources and destroys the window.
        """
        print("Closing GUI and releasing video resources.")
        if self.cap.isOpened():
            self.cap.release()
        
        # Quit the main loop FIRST, then destroy the window
        self.master.quit()
        self.master.destroy()
