import requests
import os
from typing import Tuple, Dict, Any, Union

class HamerClientAPI:
    """
    Handles network communication with the Hand Tracking API.
    
    Architectural Note: This client implements a multipart/form-data transmitter
    to sync with the server's 'process_image_file' and 'process_video_file' interfaces.
    """
    def __init__(
        self, 
        base_url: str, 
        image_timeout: Tuple[int, int] = (10, 600), 
        video_timeout: Tuple[int, int] = (10, 3600)
    ):
        """
        Initializes the client.
        
        Parameters:
        base_url (str): The base URL of the API (e.g., "http://localhost:8080").
        image_timeout (tuple): (connect, read) timeout for image uploads in seconds.
        video_timeout (tuple): (connect, read) timeout for video uploads in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.image_url = f"{self.base_url}/upload/image"
        self.video_url = f"{self.base_url}/upload/video"
        self.image_timeout = image_timeout
        self.video_timeout = video_timeout

    def _validate_params(self, hand_side: str) -> None:
        """
        Validates input parameters to prevent unnecessary network requests.
        """
        valid_sides = {"left", "right", "both"}
        if hand_side not in valid_sides:
            raise ValueError(f"Invalid hand_side: '{hand_side}'. Must be one of {valid_sides}")

    def upload_image(
        self, 
        image_path: str, 
        person_selector: Union[str, int] = "all", 
        hand_side: str = "both",
        should_render: bool = False
    ) -> Dict[str, Any]:
        """
        Sends an image file to the /upload/image endpoint with processing parameters.
        
        Parameters:
        image_path (str): Path to the image file.
        person_selector (Union[str, int]): "all", "left", "center", "right", or person index.
        hand_side (str): "both", "left", or "right".
        should_render (bool): Whether to request visualized output (default: False).
        
        Returns:
        dict: The JSON response from the server containing tracking data.
        
        Raises:
        ValueError: If parameters are invalid.
        requests.exceptions.RequestException: On connection error or HTTP error.
        """
        self._validate_params(hand_side)
        
        # Payload construction for multipart/form-data
        data_payload = {
            "person_selector": str(person_selector),
            "hand_side": hand_side,
            "should_render": str(should_render)
        }

        with open(image_path, 'rb') as f:
            # Note: Explicit MIME type ensures server handles decoding correctly
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            
            response = requests.post(
                self.image_url, 
                files=files, 
                data=data_payload,
                timeout=self.image_timeout
            )
        
        response.raise_for_status()
        return response.json()

    def upload_video(
        self, 
        video_path: str,
        person_selector: Union[str, int] = "all", 
        hand_side: str = "both",
        should_render: bool = False
    ) -> Dict[str, Any]:
        """
        Sends a video file to the /upload/video endpoint with processing parameters.
        
        Parameters:
        video_path (str): Path to the video file.
        person_selector (Union[str, int]): "all", "left", "center", "right", or person index.
        hand_side (str): "both", "left", or "right".
        should_render (bool): Whether to request visualized output (default: False).
        
        Returns:
        dict: The JSON response from the server.
        
        Raises:
        ValueError: If parameters are invalid.
        requests.exceptions.RequestException: On connection error or HTTP error.
        """
        self._validate_params(hand_side)

        data_payload = {
            "person_selector": str(person_selector),
            "hand_side": hand_side,
            "should_render": str(should_render)
        }

        with open(video_path, 'rb') as f:
            files = {'file': (os.path.basename(video_path), f, 'video/mp4')}
            
            response = requests.post(
                self.video_url, 
                files=files, 
                data=data_payload,
                timeout=self.video_timeout
            )
        
        response.raise_for_status()
        return response.json()