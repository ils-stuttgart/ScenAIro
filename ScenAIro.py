"""
ScenAIro - Main Application Controller
=======================================

This module contains the ScenAIro class, which serves as the main controller
for the application. It handles the integration between the UI layer and the
backend tools for data generation.

The ScenAIro application is designed to:
    1. Allow users to configure airport/runway parameters
    2. Generate 3D sampling points based on cone geometry
    3. Position aircraft in Microsoft Flight Simulator via SimConnect
    4. Capture screenshots and generate labeled training data

Dependencies:
    - tkinter: GUI framework
    - numpy: Numerical computations
    - SimConnect: MSFS integration
    - Custom tools: RunwayTaggingEngine, SamplingPointGenerator, etc.

Author: ScenAIro Team
"""

import tkinter as tk
from tkinter import messagebox
import numpy as np

# SimConnect integration for Microsoft Flight Simulator
from dependencies.SimConnect import *

# UI and Tool Imports
from presentation.ScenAIroUI import ScenAIroUI
from tools.RunwayTaggingEngine import RunwayTaggingEngine
from tools.SamplingPointGenerator import SamplingPointGenerator
from tools.RunwayGeometryCalculator import RunwayGeometryCalculator
from tools.GeoCoordinateProjector import GeoCoordinateProjector
from tools.AircraftPositioningAgent import AircraftPositioningAgent
from tools.RunwayCornerAnnotationStruct import RunwayCornerAnnotationStruct
from tools.WeatherAutomationAgent import WeatherAutomationAgent
from tools.SettingsManager import SettingsManager


class ScenAIro(tk.Tk):
    """
    Main application class that inherits from tk.Tk.
    
    This class acts as the controller in the MVC pattern, coordinating between
    the UI (ScenAIroUI) and the various backend tools for data generation.
    It manages application state, handles user input validation, and orchestrates
    the data generation process.
    
    Attributes:
        settings: SettingsManager singleton for configurable values
        airport: RunwayGeometryCalculator instance for runway calculations
        points: Generated 3D sampling points (numpy array)
        geo_points: Transformed geographic coordinates
        angles: Aircraft orientation angles
        tagging: RunwayTaggingEngine for overlay labeling
        coordsetter: AircraftPositioningAgent for SimConnect operations
        ui: ScenAIroUI instance for the graphical interface
    """
    
    # Class-level attribute - could be used for runway calculations if needed
    # airport = RunwayGeometryCalculator()

    def __init__(self):
        """
        Initialize the ScenAIro application.
        
        Sets up the main window, initializes the SimConnect link, creates
        instances of all backend tools, and launches the UI.
        
        Settings are loaded from the SettingsManager singleton, which provides
        configurable values for window size, paths, and other parameters.
        """
        super().__init__()
        
        # ========================================
        # Initialize Settings Manager
        # ========================================
        # The SettingsManager is a singleton that provides access to all
        # configurable application settings
        self.settings = SettingsManager()
        
        # ========================================
        # Window Configuration
        # ========================================
        # Window dimensions are now configurable via settings
        self.title("ScenAIro")
        window_width = self.settings.get("window", "width")
        window_height = self.settings.get("window", "height")
        self.geometry(f"{window_width}x{window_height}")
        self.configure(bg=self.settings.get("window", "bg_color"))

        # ========================================
        # Initialize SimConnect
        # ========================================
        # Create connection to Microsoft Flight Simulator
        sim = SimConnect()

        # ========================================
        # Application State Variables
        # ========================================
        self.airport = None              # Runway geometry calculator
        self.points = None               # Generated 3D sampling points
        self.geo_points = None           # Geographic coordinates
        self.angles = None                # Angle parameters
        self.apex = None                  # Cone apex coordinates
        self.distribution_settings = None # Point distribution settings
        
        # Aircraft orientation angles
        self.pitchMinAngle = None
        self.pitchMaxAngle = None
        self.yawMinAngle = None
        self.yawMaxAngle = None
        self.rollMinAngle = None
        self.rollMaxAngle = None
        
        # Cone geometry parameters
        self.lateral_angle_left = None
        self.lateral_angle_right = None
        self.vertical_min_angle = None
        self.vertical_max_angle = None
        self.max_distance = None
        self.aircraftOrientation = None
        self.apex_transformed = None

        # ========================================
        # Initialize Backend Tools
        # ========================================
        # RunwayTaggingEngine: Creates overlay labels on screenshots
        self.tagging = RunwayTaggingEngine()  
        
        # AircraftPositioningAgent: Positions aircraft in MSFS via SimConnect
        self.coordsetter = AircraftPositioningAgent(sim) 
        
        # RunwayCornerAnnotationStruct: Calculates runway corner annotations
        self.runwayCornerAnnotation = RunwayCornerAnnotationStruct()
        
        # SamplingPointGenerator: Generates 3D points in cone geometry
        self.pointCloudGeneration = SamplingPointGenerator()
        
        # GeoCoordinateProjector: Transforms local coords to geo coordinates
        self.transformCone = GeoCoordinateProjector()

        # ========================================
        # Create UI Layer
        # ========================================
        self.ui = ScenAIroUI(self)
        
        # Generate initial sample dataset for preview (silent=True suppresses errors)
        self.generateSampleDataset(silent=True)
        

    # ========================================
    # Input Validation Methods
    # ========================================
    
    def populateDefaultParameters(self, entry_fields, values):
        """
        Populates input fields with default values.
        
        Args:
            entry_fields: Dictionary of entry field widgets
            values: Dictionary of default values to populate
        """
        for key, value in values.items():
            entry_fields[key].delete(0, tk.END)
            entry_fields[key].insert(0, value)

    def __isFloatValue(self, value, field_name):
        """
        Validates and converts a value to float.
        
        Args:
            value: The string value to convert
            field_name: Name of the field (for error messages)
            
        Returns:
            float: The converted value
            
        Raises:
            ValueError: If the value cannot be converted to float
        """
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Invalid value for {field_name}: {value}")

    def __isIntValue(self, value, field_name):
        """
        Validates and converts a value to integer.
        
        Args:
            value: The string value to convert
            field_name: Name of the field (for error messages)
            
        Returns:
            int: The converted value
            
        Raises:
            ValueError: If the value cannot be converted to integer
        """
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid value for {field_name}: {value}")

    # ========================================
    # Sample Dataset Generation (Preview)
    # ========================================
    
    def generateSampleDataset(self, silent=False):
        """
        Generates a sample dataset based on UI parameters for preview purposes.
        
        This method reads all configuration values from the UI, validates them,
        creates the runway geometry, generates sampling points using the cone
        model, and updates the 3D visualization.
        
        Args:
            silent (bool): If True, suppresses error dialogs and only prints to console.
                          Used for live preview while typing (default: False)
                          
        Returns:
            numpy.ndarray or None: Generated 3D points if successful, None otherwise
        """
        try:
            # ------------------------------------------------
            # Step 1: Get Airport Parameters from UI
            # ------------------------------------------------
            try:
                name = self.ui.airport_entries["Airport Name"].get()
                icao = self.ui.airport_entries["ICAO Code"].get()
                runway_name = self.ui.airport_entries["Runway Name"].get()
                width = float(self.ui.airport_entries["Width"].get())
                length = float(self.ui.airport_entries["Length"].get())
                runwayHeading = float(self.ui.airport_entries["Heading"].get())
                latitude = float(self.ui.airport_entries["Latitude"].get())
                longitude = float(self.ui.airport_entries["Longitude"].get())
                altitude = float(self.ui.airport_entries["Altitude"].get())
                start_height = float(self.ui.airport_entries["Start Height"].get())
                end_height = float(self.ui.airport_entries["End Height"].get())
            except ValueError:
                raise ValueError("Please fill all fields correctly")

            # Validate required text fields
            if not all([name, icao, runway_name]):
                raise ValueError("Fields 'Airport Name', 'ICAO Code' and 'Runway Name' can't be empty")

            # Create RunwayGeometryCalculator object with airport parameters
            self.airport = RunwayGeometryCalculator(
                name, icao, runway_name, width, length, runwayHeading, latitude, longitude, altitude, start_height, end_height, {}
            )

            # ------------------------------------------------
            # Step 2: Get Cone/Point Generation Parameters
            # ------------------------------------------------
            # Read apex coordinates (the tip of the sampling cone)
            apex_x = self.__isFloatValue(self.ui.point_entries["Apex X"].get(), "Apex X")
            apex_y = self.__isFloatValue(self.ui.point_entries["Apex Y"].get(), "Apex Y")
            apex_z = self.__isFloatValue(self.ui.point_entries["Apex Z"].get(), "Apex Z")
            self.apex = (apex_x, apex_y, apex_z) 

            # Read lateral angles (left/right spread)
            self.lateral_angle_left = self.__isFloatValue(self.ui.point_entries["Lateral Angle Left"].get(), "Lateral Angle Left")
            self.lateral_angle_right = self.__isFloatValue(self.ui.point_entries["Lateral Angle Right"].get(), "Lateral Angle Right")
            
            # Read vertical angles (up/down spread)
            self.vertical_min_angle = self.__isFloatValue(self.ui.point_entries["Vertical Min Angle"].get(), "Vertical Min Angle")
            self.vertical_max_angle = self.__isFloatValue(self.ui.point_entries["Vertical Max Angle"].get(), "Vertical Max Angle")
            
            # Read maximum distance and number of points
            self.max_distance = self.__isFloatValue(self.ui.point_entries["Maximum Distance"].get(), "Maximum Distance")
            num_points = self.__isIntValue(self.ui.point_entries["Number of Points"].get(), "Number of Points")

            # ------------------------------------------------
            # Step 3: Get Aircraft Orientation Angles
            # ------------------------------------------------
            # Pitch (nose up/down), Yaw (nose left/right), Roll (bank left/right)
            self.pitchMinAngle = self.__isFloatValue(self.ui.angle_entries["Pitch Min"].get(), "Pitch Min")
            self.pitchMaxAngle = self.__isFloatValue(self.ui.angle_entries["Pitch Max"].get(), "Pitch Max")
            self.yawMinAngle = self.__isFloatValue(self.ui.angle_entries["Yaw Min"].get(), "Yaw Min")
            self.yawMaxAngle = self.__isFloatValue(self.ui.angle_entries["Yaw Max"].get(), "Yaw Max")
            self.rollMinAngle = self.__isFloatValue(self.ui.angle_entries["Bank Min"].get(), "Bank Min")
            self.rollMaxAngle = self.__isFloatValue(self.ui.angle_entries["Bank Max"].get(), "Bank Max")

            aircraftOrientationAngles = {
                "pitchMin": self.pitchMinAngle,
                "pitchMax": self.pitchMaxAngle,
                "yawMin": self.yawMinAngle,
                "yawMax": self.yawMaxAngle,
                "rollMin": self.rollMinAngle,
                "rollMax": self.rollMaxAngle
            }
            
            # ------------------------------------------------
            # Step 4: Get Distribution Settings
            # ------------------------------------------------
            self.distribution_settings = {
                "type": self.ui.distribution_var.get(), 
                "apply_x": self.ui.apply_x.get(),       
                "apply_y": self.ui.apply_y.get()        
            }

            # ------------------------------------------------
            # Step 5: Generate Points using Cone Model
            # ------------------------------------------------
            self.points, self.apex_transformed, self.aircraftOrientation = (
                SamplingPointGenerator.generateCone(
                    apex=self.apex,
                    lateral_angle_left=self.lateral_angle_left,
                    lateral_angle_right=self.lateral_angle_right,
                    vertical_min_angle=self.vertical_min_angle,
                    vertical_max_angle=self.vertical_max_angle,
                    max_distance=self.max_distance,
                    num_points=num_points,
                    heading=self.airport.runway_heading,
                    aircraftOrientationAngles=aircraftOrientationAngles,
                    distribution_settings=self.distribution_settings 
                )
            )

            # ------------------------------------------------
            # Step 6: Update 3D Visualization
            # ------------------------------------------------
            self.ui.refreshPlot(self.points, self.airport, self.apex_transformed)
            
            # Return generated points
            return self.points

        except ValueError as ve:
            # Handle validation errors
            if not silent:
                messagebox.showerror("Error", f"Invalid input: {ve}")
            else:
                # In silent mode (while typing), just log to console
                print(f"[Preview] Waiting for valid input... ({ve})")
            return None
            
        except Exception as e:
            # Handle unexpected errors
            print(f"[Error] Unexpected: {e}")
            if not silent:
                messagebox.showerror("Error", f"A really unexpected error occurred: {e}")
            return None

    # ========================================
    # Field of View (FOV) Calculation Methods
    # ========================================
    
    def __calculateVerticalFOV(self, horizontal_fov_degrees, aspect_ratio):
        """
        Calculates vertical field of view from horizontal FOV and aspect ratio.
        
        Args:
            horizontal_fov_degrees: Horizontal field of view in degrees
            aspect_ratio: Screen aspect ratio (width / height)
            
        Returns:
            float: Vertical field of view in degrees
        """
        horizontal_fov_radians = np.radians(horizontal_fov_degrees)
        vertical_fov_radians = 2 * np.arctan(np.tan(horizontal_fov_radians / 2) / aspect_ratio)
        print(f"Vertical FOV: {np.degrees(vertical_fov_radians)} degrees")
        return np.degrees(vertical_fov_radians)
    
    def __calculateHorizontalFOV(self, vertical_fov_degrees, aspect_ratio):
        """
        Calculates horizontal field of view from vertical FOV and aspect ratio.
        
        Args:
            vertical_fov_degrees: Vertical field of view in degrees
            aspect_ratio: Screen aspect ratio (width / height)
            
        Returns:
            float: Horizontal field of view in degrees
        """
        vertical_fov_radians = np.radians(vertical_fov_degrees)
        horizontal_fov_radians = 2 * np.arctan(np.tan(vertical_fov_radians / 2) * aspect_ratio)
        return np.degrees(horizontal_fov_radians)

    # ========================================
    # Data Generation (Main Export Function)
    # ========================================
    
    def generateData(self):
        """
        Generates the complete labeled dataset for ML training.
        
        This is the main data generation function that:
        1. Generates sampling points based on current configuration
        2. Transforms points to geographic coordinates
        3. Iteratively positions the aircraft in MSFS via SimConnect
        4. Captures screenshots
        5. Creates overlay labels if enabled
        6. Saves metadata alongside images
        
        The process is time-consuming as it requires the MSFS window to be
        active and waits between captures for the simulation to render.
        """
        # Check if data generation is already in progress
        if hasattr(self, "_creating_data"):
            messagebox.showwarning("Busy", "Data generation is already running!")
            return

        # Set flag: process is running
        self._creating_data = True
        
        try:
            if self.ui.labeling_var.get():
                """
                Labeled data generation mode:
                - Generate points
                - Position aircraft in MSFS
                - Capture screenshots
                - Add overlay labels to images
                """
                messagebox.showinfo("Create Data", "Creating labeled data...")

                # --- Weather Setup: Set weather conditions in MSFS ---
                try:
                    # Get desired weather from UI (defined in ScenAIroUI.py)
                    selected_weather = self.ui.weather_var.get()
                    
                    # Initialize and run the weather agent
                    weather_bot = WeatherAutomationAgent()
                    success = weather_bot.set_weather(selected_weather)
                    
                    if not success:
                        print("[Warning] Could not set weather automatically.")
                        # Optional: messagebox.showwarning("Weather", "Could not set weather automatically")
                        
                except Exception as w_err:
                    print(f"[Error] Weather script error: {w_err}")

                # --- Step 1: Generate Points ---
                generated_points = self.generateSampleDataset(silent=False)
                if generated_points is None or len(generated_points) == 0:
                    raise ValueError("No generated points available. Please check input values.")

                # --- Step 2: Prepare Metadata ---
                center_lat = self.airport.runway_center["latitude"]
                center_lon = self.airport.runway_center["longitude"]
                center_alt = self.airport.runway_center["altitude"]
                heading = 0  # Base heading (relative to runway)

                # --- Step 3: Transform to Geographic Coordinates ---
                geo_points = GeoCoordinateProjector.transform_points(
                    points=generated_points,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    center_alt=center_alt,
                    heading=heading
                )

                if geo_points is None or len(geo_points) == 0:
                    raise ValueError("Failed to transform points to geographic coordinates.")

                # --- Step 4: Configure Output Settings ---
                # Screenshot path is now configurable via settings
                screenshot_path = self.settings.get("paths", "screenshot_path")
                
                # Read checkbox options
                excludeImg = self.ui.labeling_exclImg.get()         # Skip image creation
                createOverlay = self.ui.validation_var.get()        # Add overlay labels

                # Calculate runway corners for annotation
                corners = self.airport.calculateRunwayCorners()
                
                # Get simulation time settings
                setSimHour = self.__isIntValue(self.ui.time_entries["Hours"].get(), "Hours")
                setSimMin  = self.__isIntValue(self.ui.time_entries["Minutes"].get(), "Minutes")

                # Prepare airport metadata
                airport_metadata = {
                    "name": self.airport.name,
                    "icao_code": self.airport.icao_code,
                    "runway_name": self.airport.runway_name,
                    "runway_width": self.airport.runway_width,
                    "runway_length": self.airport.runway_length,
                    "runway_heading": self.airport.runway_heading,
                    "runway_center": self.airport.runway_center,
                    "start_height": self.airport.start_height,
                    "end_height": self.airport.end_height,
                }
                
                dist_settings = getattr(self, "distribution_settings", 
                                       {"type": "Uniform", "apply_x": False, "apply_y": False})

                # Prepare cone/trajectory metadata
                cone_metadata = {
                    "apex": self.apex,
                    "lateral_angle_left": self.lateral_angle_left,
                    "lateral_angle_right": self.lateral_angle_right,
                    "vertical_min_angle": self.vertical_min_angle,
                    "vertical_max_angle": self.vertical_max_angle,
                    "max_distance": self.max_distance,
                    "number_of_points": len(generated_points),
                    "distribution": dist_settings
                }

                daytime = {"hours": setSimHour, "minutes": setSimMin}
                
                # Store selected weather for metadata
                selected_weather = self.ui.weather_var.get()

                # --- Step 5: Initialize SimConnect for Data Collection ---
                sim = SimConnect()
                coord_setter = AircraftPositioningAgent(sim)

                # --- Step 6: Iterate through all generated points ---
                for i, (geo_point, generated_point) in enumerate(zip(geo_points[1:], generated_points)):
                    # Extract geographic coordinates and convert altitude to feet
                    latitude, longitude, altitude = map(float, geo_point)
                    altitude *= 3.28084  # Convert meters to feet

                    # Get aircraft orientation for this point
                    pitch = float(self.aircraftOrientation["pitch"][i])
                    roll = float(self.aircraftOrientation["roll"][i])
                    yaw_offset = float(self.aircraftOrientation["yaw"][i])

                    # Configure camera/screenshot settings
                    # All these values are now configurable via settings
                    vertical_fov_radians = self.settings.get("camera", "vertical_fov_radians")
                    vertical_fov_degrees = np.degrees(vertical_fov_radians)
                    screen_width = self.settings.get("screen", "width")
                    screen_height = self.settings.get("screen", "height")
                    aspectRatio = screen_width / screen_height
                    runwayHeading = self.airport.runway_heading
                    
                    # Calculate horizontal FOV from vertical FOV and aspect ratio
                    horizontal_fov_degrees = self.__calculateHorizontalFOV(vertical_fov_degrees, aspectRatio)
                    
                    # Calculate final heading (runway direction - camera offset)
                    final_heading = (runwayHeading - 180) - yaw_offset

                    # --- Step 7: Position Aircraft and Capture Screenshot ---
                    screenshot_name = coord_setter.positionAircraftInSimAndTakeScreenshot(
                        latitude, longitude, altitude, pitch, final_heading, roll, 
                        screenshot_path, screen_width, screen_height, setSimHour, setSimMin, excludeImg,
                    )
                    
                    # --- Step 8: Calculate Runway Corner Annotations ---
                    runway_annotation = RunwayCornerAnnotationStruct()
                    structured_objects = runway_annotation.calculateAirplane2RunwayCornerStructure(
                        generated_point, corners, (pitch, yaw_offset, roll), runwayHeading, center_alt
                    )

                    # --- Step 9: Create Overlay Labels (if enabled) ---
                    # Skip image tagging if:
                    # 1. excludeImg is True (no images wanted)
                    # 2. OR createOverlay is False (no overlays wanted)
                    skip_tagging_image = excludeImg or not createOverlay
                    
                    print(f"Pitch: {pitch}, Yaw: {yaw_offset}, Roll: {roll}")

                    self.tagging.doOverlayLabelsOnImage(
                        image_path=f"{screenshot_path}\\{screenshot_name}.png",
                        output_path=f"{screenshot_path}\\tagged_{screenshot_name}.png",
                        structured_objects=structured_objects,
                        horizontal_fov_degrees=horizontal_fov_degrees,
                        vertical_fov_degrees=vertical_fov_degrees,
                        screen_width=screen_width,
                        screen_height=screen_height,
                        cam_pitch= pitch,
                        cam_yaw= yaw_offset,
                        cam_roll= roll,
                        airport_data = airport_metadata,
                        cone_data= cone_metadata, 
                        geo_point= geo_point,
                        generated_point= generated_point,
                        daytime=daytime,
                        weather_data=selected_weather,
                        excludeImg= skip_tagging_image 
                    )

                messagebox.showinfo("Success", "Data and screenshots created successfully.")

            else:
                # Unlabeled data generation mode (basic point generation without MSFS)
                messagebox.showinfo("Create Data", "Creating data without labeling...")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            print(e)

        finally:
            # Clean up flag when done
            if hasattr(self, "_creating_data"):
                del self._creating_data
        

