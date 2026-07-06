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
import time
import copy
from pyproj import Geod
from dependencies.SimConnect import SimConnect
from tools.AircraftPositioningAgent import AircraftPositioningAgent
from tools.WeatherAutomationAgent import WeatherAutomationAgent
from tools.SettingsManager import SettingsManager
from tools.RunwayTaggingEngine import RunwayTaggingEngine
from tools.RunwayGeometryCalculator import RunwayGeometryCalculator
from tools.RunwayCornerAnnotationStruct import RunwayCornerAnnotationStruct


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
        self.settings = SettingsManager()
        
        # RunwayTaggingEngine: Creates overlay labels on screenshots
        self.tagging = RunwayTaggingEngine() 
                

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
        
        with open(self.file_path, "r", encoding="utf-8-sig") as f:
            self.metadata = json.load(f)
        return self.metadata

    def _ensure_loaded(self):
        """Helper method: loads metadata if not already loaded."""
        if not self.metadata:
            self.load_metadata()

    def get_image_info(self):
        """
        Get the first entry from the 'images' block.
        If width or height from JSON are invalid (<= 0) or missing, use 2560x1440.
        
        Returns:
            dict: Image information dictionary with valid width/height
            
        Raises:
            ValueError: If no 'images' entries are found
        """
        self._ensure_loaded()
        images = self.metadata.get("images", [])
        if not images:
            raise ValueError("No 'images' entries found in metadata.")
        
        img_info = images[0].copy()  # Work on a copy to avoid modifying original
        
        # Versuche die Werte aus der JSON zu lesen. Falls nicht vorhanden, nimm standardmäßig 0.
        width = img_info.get("width", 0)
        height = img_info.get("height", 0)
        
        # Wenn die Werte 0 oder kleiner sind, setze sie hart auf deine Wunschwerte
        if width <= 0:
            img_info["width"] = 2560
            print(f"[MetadataFileReader] Invalid or missing width ({width}) from JSON, using fallback: 2560")
        
        if height <= 0:
            img_info["height"] = 1440
            print(f"[MetadataFileReader] Invalid or missing height ({height}) from JSON, using fallback: 1440")
        
        return img_info
    
    def get_fov_info(self, width, height):
        """
        Get FOV Values from Settings and convert to degrees.
        """
        vertical_fov_radians = self.settings.get("camera", "vertical_fov_radians")
        
        # ODER du nutzt dein Property aus dem SettingsManager:
        # vertical_fov_radians = self.settings.vertical_fov_radians
        
        # 2. In Grad umwandeln
        vertical_fov_degrees = np.degrees(vertical_fov_radians)
        
        # 3. Horizontales FOV über die Aspect Ratio berechnen (genau wie in ScenAIro.py)
        aspect_ratio = width / height
        horizontal_fov_radians = 2 * np.arctan(np.tan(vertical_fov_radians / 2) * aspect_ratio)
        horizontal_fov_degrees = np.degrees(horizontal_fov_radians)
        
        return horizontal_fov_degrees, vertical_fov_degrees

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
    
    def _get_airport_name(self):
        """
        Extract airport name from metadata.
        
        Returns:
            str or int: Airport identifier from runway_data.name
        """
        self._ensure_loaded()
        runway_data = self.metadata.get("runway_data", {})
        airport_name = runway_data.get("name")
        return airport_name

    def _get_airport_position(self):
        """
        Extract airport position from metadata.
        
        Parses 'position_of_aircraft' field from the JSON.
        
        Returns:
            tuple: (latitude, longitude, altitude) as floats
        """
        self._ensure_loaded()
        runway_data = self.metadata.get("runway_data", {})
        runway_coordinates = runway_data.get("runway_center")
        lat = runway_coordinates.get("latitude")
        lon = runway_coordinates.get("longitude")
        alt = runway_coordinates.get("altitude")
        print("Coordinates of runway", lat, lon, alt)
        return lat, lon, alt
    
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

    def check_and_wait_for_conditions(self, prev_airport, prev_weather, set_weather=True):
        """
        Check weather and airport changes between consecutive JSON files.
        
        Applies wait times based on changes:
        - Weather changed: set new weather, wait 1 second
        - Weather same: wait 1 second
        - Airport changed: wait 4 seconds
        - Airport same: wait 1 second
        
        Args:
            prev_airport: Previous airport identifier (or None for first file)
            prev_weather: Previous weather value (or None for first file)
            set_weather: Whether to set weather based on metadata
            
        Returns:
            tuple: (current_airport, current_weather, weather_set)
        """
        self._ensure_loaded()
        
        current_airport = self._get_airport_name()
        current_weather = self._get_weather()
        weather_set = None
        
        # Handle Heavy Rain special case
        if current_weather == "Heavy Rain":
            current_weather = "Light Thunderstorm"
        
        # Check if weather changed
        weather_changed = (prev_weather is None or current_weather != prev_weather)
        
        # Check if airport changed
        airport_changed = (prev_airport is None or current_airport != prev_airport)
        
        # Apply wait times and weather setting
        if weather_changed and set_weather and current_weather:
            try:
                weather_bot = WeatherAutomationAgent()
                success = weather_bot.set_weather(current_weather)
                if success:
                    weather_set = current_weather
                    print(f"[MetadataFileReader] Weather changed: {prev_weather} -> {current_weather}")
            except Exception as w_err:
                print(f"[MetadataFileReader] Weather script error: {w_err}")
            time.sleep(1)
        elif not weather_changed:
            time.sleep(1)
        
        if airport_changed:
            print(f"[MetadataFileReader] Airport changed: {prev_airport} -> {current_airport}, waiting 4s")
            time.sleep(4)
        else:
            print(f"[MetadataFileReader] Airport same: {current_airport}, waiting 1s")
            time.sleep(1)
        
        return current_airport, current_weather, weather_set

    # ---------------------------------------------------------
    # Trasform aircraft position into cartesian coordinates
    # and calculate relative position of runway
    # ---------------------------------------------------------

    def _transform_aircraft_LLA_2_cartesian(self):
        lat_aircraft, lon_aircraft, alt_aircraft = self._get_aircraft_position()
        lat_runway, lon_runway, alt_runway = self._get_airport_position()
        heading_runway = self._get_runway_heading()
        
        geod = Geod(ellps='WGS84')
        
        azimuth_forward, _, distance_2d = geod.inv(
            lon_runway, lat_runway,
            lon_aircraft, lat_aircraft
        )
        
        relative_angle_deg = azimuth_forward #- heading_runway
        relative_angle_rad = np.radians(relative_angle_deg)
        
        x = distance_2d * np.cos(relative_angle_rad)
        y = distance_2d * np.sin(relative_angle_rad)
        z = alt_aircraft - alt_runway
        
        # Calculate ground distance using Pythagorean theorem (x, y plane)
        distance_ground = float(np.sqrt(x**2 + y**2))

        # Height difference (z component)
        altitude_difference = float(z)

        distance_ground = round(distance_ground, 2)
        altitude_difference = round(altitude_difference, 2)
        
        print("This is the distance from the runway to the aircraft!!!", " x: ", x, " y: ", y, " z: ", z)
        print("This is the distance:", " ground distance: ", distance_ground, " altitude difference: ", altitude_difference)
        
        return np.array([x, y, z])
        
    # ---------------------------------------------------------
    # Image Generation from JSON (using MSFS + AircraftPositioningAgent)
    # ---------------------------------------------------------

    def generate_and_save_annotation(
        self,
        use_sim=True,
        set_weather=True,
        output_annotation_folder=None,
        excludeImg=False,
        current_active_weather=None,
        sim_connection=None,
        coord_setter=None
    ):
        """
        Generate annotation from metadata, take screenshot, and save merged JSON.
        
        This method:
        1. Loads the existing JSON metadata
        2. Extracts all relevant info
        3. Sets the aircraft position and takes a screenshot
        4. Calculates runway corner annotations (COCO format with bbox & segmentation)
        5. Saves a new JSON file (in a new folder) containing the original data
           plus the generated annotation data
        
        Args:
            use_sim: Whether to use MSFS for screenshot generation
            set_weather: Whether to set weather based on metadata
            output_annotation_folder: Folder to save the merged JSON file.
                If None, creates a 'annotations' subfolder in the same directory
                as the source JSON file.
            excludeImg: If True, skip image creation (only generate annotations)
            
        Returns:
            tuple: (base_image_path, output_json_path, weather_set)
        """
        json_basename = os.path.splitext(os.path.basename(self.file_path))[0]
        source_dir = os.path.dirname(self.file_path)
        
        # Determine output directory for annotations
        if output_annotation_folder is None:
            output_annotation_folder = os.path.join(source_dir, "annotations")
        os.makedirs(output_annotation_folder, exist_ok=True)
        
        
        # Use the base screenshot directory for images
        use_screenshot_dir = self.screenshot_dir
        weather_set = current_active_weather
        
        # 1-3: Load, extract, set aircraft, take screenshot
        if use_sim:
            try:
                # Weather setup
                if set_weather:
                    weather_condition = self._get_weather()
                    if weather_condition == "Heavy Rain":
                        weather_condition = "Light Thunderstorm"
                        
                    # Jetzt greift der smarte Check wirklich!
                    if weather_condition and weather_condition != weather_set:
                        try:
                            weather_bot = WeatherAutomationAgent()
                            success = weather_bot.set_weather(weather_condition)
                            if success:
                                weather_set = weather_condition # Update tracking
                                print(f"[MetadataFileReader] Weather set to: {weather_condition}")
                        except Exception as w_err:
                            print(f"[MetadataFileReader] Weather script error: {w_err}")
                
                # Load aircraft data
                lat, lon, alt = self._get_aircraft_position()
                runway_heading = self._get_runway_heading()
                hours, minutes = self._get_daytime()
                pitch, yaw, roll = self._get_aircraft_orientation()
                
                heading = runway_heading - 180.0 + yaw
                
                # Position aircraft and take screenshot
                if coord_setter is not None:
                    agent = coord_setter
                else:
                    sim = sim_connection if sim_connection is not None else SimConnect()
                    agent = AircraftPositioningAgent(sim)
                
                alt_feet = alt * 3.28084
                
                img_info = self.get_image_info()
                width = int(img_info.get("width", 2560))
                height = int(img_info.get("height", 1440))
                
                screenshot_name = agent.positionAircraftInSimAndTakeScreenshot(
                    lat, lon, alt_feet, pitch, heading, roll,
                    use_screenshot_dir, width, height, hours, minutes,
                    excludeImg=False,
                    custom_filename=json_basename
                )
                
                #time.sleep(1)
                
                base_image_path = os.path.join(use_screenshot_dir, f"{screenshot_name}.png")
                
            except Exception as e:
                print(f"[MetadataFileReader] Error with SimConnect: {e}")
                base_image_path = os.path.join(use_screenshot_dir, f"{json_basename}.png")
        else:
            print("[MetadataFileReader] use_sim=False - skipping MSFS.")
            base_image_path = os.path.join(use_screenshot_dir, f"{json_basename}.png")
            img_info = self.get_image_info()
            width = int(img_info.get("width", 2560))
            height = int(img_info.get("height", 1440))
            hours, minutes = self._get_daytime()
            runway_heading = self._get_runway_heading()
            lat, lon, alt = self._get_aircraft_position()
            pitch, yaw, roll = self._get_aircraft_orientation()
            pitch = 0.0
            roll = 0.0
            heading = runway_heading - 180.0 + yaw
        
        # Load FOV info
        horizontal_fov, vertical_fov = self.get_fov_info(width, height)
        
       
        # 4: Calculate annotations
        runway_data = self.metadata["runway_data"]
        cone_data = self.metadata.get("landing_approach_cone", {})
        
        airport = RunwayGeometryCalculator(
            name=runway_data["name"],
            icao_code=runway_data["icao_code"],
            runway_name=runway_data["runway_name"],
            runway_width=float(runway_data["runway_width"]),
            runway_length=float(runway_data["runway_length"]),
            runway_heading=float(runway_heading),
            center_lat=float(runway_data["runway_center"]["latitude"]),
            center_long=float(runway_data["runway_center"]["longitude"]),
            center_alt=float(runway_data["runway_center"]["altitude"]),
            start_height=float(runway_data["start_height"]),
            end_height=float(runway_data["end_height"]),
            runway_attributes={}
        )
        corners = airport.calculateRunwayCorners()
        
        # Calculate local coordinates
        
        generated_point = self._transform_aircraft_LLA_2_cartesian()                                # It's already aligned, not as the original generated point which is not yet tilted by the runway heading
        local_x, local_y, local_z = generated_point[0], generated_point[1], generated_point[2]
        
        # Calculate runway corner annotation structure
        runway_annotation = RunwayCornerAnnotationStruct()                         
        structured_objects = runway_annotation.calculateAirplane2RunwayCornerStructure(
            point=generated_point,
            runway_corners=corners,
            angles=(pitch, yaw, roll),
            runwayHeading=runway_heading,
            centerHeight=float(runway_data["runway_center"]["altitude"])
        )
        
        # Get the first structured object (list contains one item)
        so = structured_objects[0]
        
        # Corner tuples from StructuredObject (A, B, C, D are tuples of 3 floats each)
        corner_A, corner_B, corner_C, corner_D = so.A, so.B, so.C, so.D
        
        # Calculate pixel projections for each corner (clockwise: A->B->C->D)       # should same as otherwise
        pixel_coords = []
        for point in [corner_A, corner_B, corner_C, corner_D]:
            rotated_point = self.tagging.rotate3DPoint(point, pitch, yaw, roll)
            x_pixel, y_pixel = self.tagging.calculatePixelCoordinates(
                rotated_point, horizontal_fov, vertical_fov, width, height
            )
            pixel_coords.extend([int(x_pixel), int(y_pixel)])
        
        # Calculate bounding box from corner points
        pixel_x = pixel_coords[0::2]
        pixel_y = pixel_coords[1::2]
        bbox_x = min(pixel_x)
        bbox_y = min(pixel_y)
        bbox_w = max(pixel_x) - bbox_x
        bbox_h = max(pixel_y) - bbox_y
        
        bbox = [bbox_x, bbox_y, bbox_w, bbox_h]
        area = bbox_w * bbox_h
        
        # 5: Build merged annotation JSON with original data + COCO-style annotations
        merged_data = copy.deepcopy(self.metadata)
        
        # COCO-format annotation entry
        coco_annotation = {
            "id": 0,
            "image_id": f"{json_basename}.png",
            "category_id": 1,
            "bbox": bbox,
            "segmentation": [pixel_coords],
            "area": area,
            "iscrowd": 0,
        }
        
        # Ensure images array uses correct format
        merged_data["images"] = [
            {
                "file_name": f"{json_basename}.png",
                "id": f"{json_basename}.png",
                "width": width,
                "height": height,
            }
        ]
        
        # Set annotations array
        merged_data["annotations"] = [coco_annotation]
        
        # Ensure categories exist
        if "categories" not in merged_data:
            merged_data["categories"] = [
                {"id": 1, "name": "runway", "supercategory": "infrastructure"}
            ]
        
        # Add/update metadata fields                                                    
        merged_data["runway_data"] = runway_data
        if cone_data:
            merged_data["landing_approach_cone"] = cone_data
        merged_data["position_of_aircraft"] = [lat, lon, alt]
        merged_data["distance_aircraft_2_runway"] = {
            "ground_distance_in_meters": round(float(np.sqrt(local_x**2 + local_y**2)), 2),
            "altitude_difference_in_meters": round(float(local_z), 2),
        }
        merged_data["aircraft_orientation"] = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        }
        merged_data["daytime"] = {"hours": hours, "minutes": minutes}
        weather_data = self._get_weather()
        if weather_data:
            merged_data["weather"] = weather_data
        
        # Save merged JSON to annotation folder
        output_json_path = os.path.join(output_annotation_folder, f"{json_basename}.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=4, ensure_ascii=False)
        
        print(f"[MetadataFileReader] Annotated JSON saved: {output_json_path}")
        print(f"[MetadataFileReader] Process finished for {json_basename}")
        
        return base_image_path, output_json_path, weather_set
  
        

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
        output_jsons = []
        current_weather = None  # Track current weather to avoid unnecessary changes
        current_airport = None  # Track current airport for wait time logic
        
        main_sim_connection = None
        main_coord_setter = None
        if use_sim:
            try:
                main_sim_connection = SimConnect()
                main_coord_setter = AircraftPositioningAgent(main_sim_connection) 
                print("[MetadataFileReader] Master-SimConnect-Connection & Agent successfully built.")
            except Exception as e:
                print(f"[ERROR] Couldn't setup initial SimConnect connection: {e}")
                return [], []

        for json_file in json_files:
            try:
                print(f"[MetadataFileReader] Processing: {json_file}")

                reader = MetadataFileReader(json_file, screenshot_dir=folder_path)
                reader.load_metadata()

                # Check for weather/airport changes and apply wait times
                current_airport, current_weather, weather_set = reader.check_and_wait_for_conditions(
                    current_airport, current_weather, set_weather=set_weather
                )

                # Use generate_and_save_annotation for full pipeline
                out_path, json_path, weather_set = reader.generate_and_save_annotation(
                    use_sim=use_sim,
                    set_weather=set_weather,
                    output_annotation_folder=os.path.join(folder_path, "annotations"),
                    excludeImg=False,
                    current_active_weather=current_weather,
                    sim_connection=main_sim_connection,
                    coord_setter=main_coord_setter 
                )
                
                # Update current weather tracking if weather was set
                if weather_set:
                    current_weather = weather_set

                output_images.append(out_path)
                output_jsons.append(json_path)
                print(f"[MetadataFileReader] Completed: {out_path}")
                print(f"[MetadataFileReader] Annotation: {json_path}")

            except Exception as e:
                print(f"[MetadataFileReader] Error processing {json_file}: {e}")

        print(f"[MetadataFileReader] All files processed ({len(output_images)} images generated, {len(output_jsons)} annotations saved).")
        return output_images, output_jsons
