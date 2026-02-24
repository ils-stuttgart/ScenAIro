"""
Metadata File Reader Module

This module provides functionality for reading ScenAIro/COCO-style JSON metadata
files and regenerating screenshots using the AircraftPositioningAgent and
Microsoft Flight Simulator (MSFS).
"""

import os
import json
import cv2
import numpy as np

from dependencies.SimConnect import SimConnect
from tools.AircraftPositioningAgent import AircraftPositioningAgent
from tools.WeatherAutomationAgent import WeatherAutomationAgent


class MetadataFileReader:
    """
    Reader for ScenAIro/COCO-style JSON metadata files.
    
    Parses metadata files and can regenerate screenshots by repositioning
    the aircraft in MSFS using the AircraftPositioningAgent.
    """

    def __init__(self, file_path, screenshot_dir=None):
        """
        Initialize the metadata reader.
        
        Args:
            file_path: Path to the JSON metadata file
            screenshot_dir: Base directory for new screenshots.
                           If None, uses the directory of the JSON file.
        """
        self.file_path = file_path
        self.metadata = {}
        self.screenshot_dir = screenshot_dir or os.path.dirname(file_path)

    # ---------------------------------------------------------
    # Loading and Accessing Metadata
    # ---------------------------------------------------------

    def load_metadata(self):
        """
        Load JSON metadata from the configured file path.
        
        Returns:
            dict: The loaded metadata dictionary
            
        Raises:
            FileNotFoundError: If the metadata file does not exist
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Metadata JSON not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return self.metadata

    def _ensure_loaded(self):
        """Helper method: loads metadata if not already loaded."""
        if not self.metadata:
            self.load_metadata()

    def get_image_info(self):
        """
        Get the first entry from the 'images' block.
        
        Returns:
            dict: Image information dictionary
            
        Raises:
            ValueError: If no 'images' entries are found
        """
        self._ensure_loaded()
        images = self.metadata.get("images", [])
        if not images:
            raise ValueError("No 'images' entries found in metadata.")
        return images[0]

    def get_annotations(self):
        """
        Get the list of annotations from metadata.
        
        Returns:
            list: List of annotation dictionaries
        """
        self._ensure_loaded()
        return self.metadata.get("annotations", [])

    # ---------------------------------------------------------
    # Helper Functions for Metadata to Simulator Parameters
    # ---------------------------------------------------------

    def _get_aircraft_position(self):
        """
        Extract aircraft position from metadata.
        
        Parses 'position_of_aircraft' field from the JSON.
        
        Returns:
            tuple: (latitude, longitude, altitude) as floats
            
        Raises:
            ValueError: If the field is missing or has invalid format
        """
        self._ensure_loaded()
        pos = self.metadata.get("position_of_aircraft")
        if pos is None:
            raise ValueError("Field 'position_of_aircraft' missing in metadata.")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            lat, lon, alt = map(float, pos[:3])
            return lat, lon, alt
        raise ValueError(f"Invalid format for 'position_of_aircraft': {pos}")
    
    def _get_aircraft_orientation(self):
        """
        Extract aircraft orientation angles from metadata.
        
        Returns:
            tuple: (pitch, yaw, roll) as integers in degrees
        """
        aircraft_orientation = self.metadata.get("aircraft_orientation", {})
        pitch = int(aircraft_orientation.get("pitch"))
        yaw = int(aircraft_orientation.get("yaw"))
        roll = int(aircraft_orientation.get("roll"))
        return pitch, yaw, roll

    def _get_runway_heading(self):
        """
        Extract runway heading from metadata.
        
        Returns:
            float: Runway heading in degrees
            
        Raises:
            ValueError: If runway_data or runway_heading is missing
        """
        self._ensure_loaded()
        runway_data = self.metadata.get("runway_data") or self.metadata.get("runway", {})
        if not runway_data:
            raise ValueError("Field 'runway_data' missing in metadata.")
        heading = runway_data.get("runway_heading")
        if heading is None:
            raise ValueError("Field 'runway_heading' missing in 'runway_data'.")
        return float(heading)

    def _get_daytime(self):
        """
        Extract daytime information from metadata.
        
        Returns:
            tuple: (hours, minutes) as integers
        """
        self._ensure_loaded()
        daytime = self.metadata.get("daytime", {})
        hours = int(daytime.get("hours", 12))
        minutes = int(daytime.get("minutes", 0))
        return hours, minutes

    def _get_weather(self):
        """
        Extract weather information from metadata.
        
        Returns:
            str or None: Weather condition string (e.g., "Clear Skies", "Rain")
                        Returns None if no weather data is present.
        """
        self._ensure_loaded()
        weather = self.metadata.get("weather")
        return weather

    # ---------------------------------------------------------
    # Image Generation from JSON (using MSFS + AircraftPositioningAgent)
    # ---------------------------------------------------------

    def generate_image_from_metadata(
        self,
        use_sim=True,
        output_path=None,
        draw_bbox=True,
        draw_segmentation=True,
        line_thickness=2,
        set_weather=True,
        current_weather=None,
    ):
        """
        Generate an image based on the JSON metadata.
        
        When use_sim=True:
            - Uses SimConnect and AircraftPositioningAgent to reposition
              the aircraft and capture a new screenshot.
        When use_sim=False:
            - Creates an empty canvas with matching resolution.
        
        Args:
            use_sim: Whether to use MSFS and AircraftPositioningAgent
            output_path: Target path for the generated image.
                        If None, saves as '<json_basename>_from_json.png'
                        in the same directory as the JSON file.
            draw_bbox: Whether to draw bounding boxes
            draw_segmentation: Whether to draw segmentation polygons
            line_thickness: Line thickness for overlays
            set_weather: Whether to set weather in MSFS (default: True)
            current_weather: The currently set weather (to avoid re-setting same weather)
            
        Returns:
            tuple: (path to generated image, weather that was set or None)
            
        Raises:
            IOError: If the image cannot be saved
        """
        self._ensure_loaded()

        # --- Extract image information from JSON ---
        img_info = self.get_image_info()
        width = int(img_info.get("width", 2560))
        height = int(img_info.get("height", 1440))

        # Determine output path
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = os.path.join(self.screenshot_dir, f"{base_name}_from_json.png")

        # -----------------------------------------
        # Generate base image using MSFS
        # -----------------------------------------
        weather_set = None
        if use_sim:
            try:
                # --- Weather Setup: Set weather conditions in MSFS ---
                if set_weather:
                    weather_condition = self._get_weather()
                    
                    # Only set weather if it's different from current weather
                    if weather_condition and weather_condition != current_weather:
                        try:
                            weather_bot = WeatherAutomationAgent()
                            success = weather_bot.set_weather(weather_condition)
                            
                            if success:
                                weather_set = weather_condition
                                print(f"[MetadataFileReader] Weather set to: {weather_condition}")
                            else:
                                print(f"[MetadataFileReader] Warning: Could not set weather to '{weather_condition}'")
                        except Exception as w_err:
                            print(f"[MetadataFileReader] Weather script error: {w_err}")
                    elif weather_condition == current_weather:
                        print(f"[MetadataFileReader] Weather already set to '{current_weather}', skipping...")
                        weather_set = current_weather
                    else:
                        print("[MetadataFileReader] No weather data in JSON, skipping weather setup.")
                
                lat, lon, alt = self._get_aircraft_position()
                runway_heading = self._get_runway_heading()
                hours, minutes = self._get_daytime()
                pitch, yaw, roll = self._get_aircraft_orientation()

                pitch = 0.0
                roll = 0.0
                # As in Data-Generator: flight direction opposite to runway -> Heading - 180
                heading = runway_heading - 180.0 + yaw

                # Initialize SimConnect and AircraftPositioningAgent
                sim = SimConnect()
                coord_setter = AircraftPositioningAgent(sim)

                # Conversion from meters to feet for MSFS
                alt *= 3.28084

                # Screenshot path is the directory of the JSON file
                screenshot_name = coord_setter.positionAircraftInSimAndTakeScreenshot(
                    lat,
                    lon,
                    alt,
                    pitch,
                    heading,
                    roll,
                    self.screenshot_dir,
                    width,
                    height,
                    hours,
                    minutes,
                    excludeImg=False  # We want actual images
                )

                base_image_path = os.path.join(self.screenshot_dir, f"{screenshot_name}.png")
                image = cv2.imread(base_image_path)
                if image is None:
                    print("[MetadataFileReader] Warning: Screenshot could not be loaded, "
                          "creating empty canvas.")
                    image = np.zeros((height, width, 3), dtype=np.uint8)
            except Exception as e:
                print(f"[MetadataFileReader] Error with SimConnect/AircraftPositioningAgent: {e}")
                print("[MetadataFileReader] Fallback: empty canvas without sim.")
                image = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # No sim: create empty image
            print("[MetadataFileReader] use_sim=False - creating empty canvas.")
            image = np.zeros((height, width, 3), dtype=np.uint8)

        # -----------------------------------------
        # Draw annotations on the image
        # -----------------------------------------
        annotations = self.get_annotations()
        if not annotations:
            print("[MetadataFileReader] Warning: No annotations found in JSON.")

        # Color: BGR (Green)
        color = (0, 255, 0)

        for anno in annotations:
            # Draw bounding box
            """
            if draw_bbox and "bbox" in anno:
                bbox = anno["bbox"]
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x, y, w, h = bbox
                    x1 = int(round(x))
                    y1 = int(round(y))
                    x2 = int(round(x + w))
                    y2 = int(round(y + h))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

            # Draw segmentation (COCO-Style)
            if draw_segmentation and "segmentation" in anno:
                seg_list = anno["segmentation"]

                # Case 1: [[x1,y1,x2,y2,...]]
                if isinstance(seg_list, list) and len(seg_list) > 0 and isinstance(seg_list[0], list):
                    segs = seg_list
                # Case 2: [x1,y1,x2,y2,...] (flat list)
                elif isinstance(seg_list, list):
                    segs = [seg_list]
                else:
                    segs = []

                for seg in segs:
                    if not isinstance(seg, list) or len(seg) < 4:
                        continue
                    pts = np.array(
                        list(zip(seg[0::2], seg[1::2])),
                        dtype=np.int32
                    ).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=line_thickness)
            """

        # -----------------------------------------
        # Save the image
        # -----------------------------------------
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        success = cv2.imwrite(output_path, image)
        if not success:
            raise IOError(f"Failed to save image: {output_path}")

        print(f"[MetadataFileReader] Image generated from JSON: {output_path}")
        return output_path, weather_set

    # ---------------------------------------------------------
    # Folder Processing: Process All JSON Files
    # ---------------------------------------------------------

    def process_folder(self, folder_path, use_sim=True, set_weather=True):
        """
        Process all JSON files in a folder and generate images for each.
        
        This method efficiently handles weather settings by only changing
        the weather when it differs from the previously set weather.
        
        Args:
            folder_path: Path to the folder containing JSON files
            use_sim: Whether to use MSFS and AircraftPositioningAgent
            set_weather: Whether to set weather based on JSON metadata
            
        Returns:
            list: Paths to the generated images
            
        Raises:
            NotADirectoryError: If the folder path is not a directory
            FileNotFoundError: If no JSON files are found in the folder
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Folder not found: {folder_path}")

        json_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".json")
        ]

        if not json_files:
            raise FileNotFoundError("No JSON files found in folder.")

        print(f"[MetadataFileReader] Found {len(json_files)} JSON files in {folder_path}")

        output_images = []
        current_weather = None  # Track current weather to avoid unnecessary changes

        for json_file in json_files:
            try:
                print(f"[MetadataFileReader] Processing: {json_file}")

                reader = MetadataFileReader(json_file, screenshot_dir=folder_path)
                reader.load_metadata()

                out_path, weather_set = reader.generate_image_from_metadata(
                    use_sim=use_sim,
                    set_weather=set_weather,
                    current_weather=current_weather
                )
                
                # Update current weather tracking if weather was set
                if weather_set:
                    current_weather = weather_set

                output_images.append(out_path)
                print(f"[MetadataFileReader] Completed: {out_path}")

            except Exception as e:
                print(f"[MetadataFileReader] Error processing {json_file}: {e}")

        print(f"[MetadataFileReader] All files processed ({len(output_images)} images generated).")
        return output_images
