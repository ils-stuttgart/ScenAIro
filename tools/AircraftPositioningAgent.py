"""
AircraftPositioningAgent - SimConnect Integration for Aircraft Control
=======================================================================

This module provides the AircraftPositioningAgent class which handles
communication with Microsoft Flight Simulator (MSFS) via SimConnect.

The agent can:
    - Set aircraft position (latitude, longitude, altitude)
    - Set aircraft orientation (pitch, heading, bank/roll)
    - Control simulation time (hours, minutes)
    - Freeze aircraft position for consistent screenshots
    - Capture screenshots of the simulation window

Dependencies:
    - pygetwindow: For finding and activating MSFS window
    - PIL: For image processing
    - mss: For fast screen capture
    - SimConnect: For MSFS communication
    - numpy: For numerical operations

Author: ScenAIro Team
"""

import pygetwindow
from PIL import Image
from dependencies.SimConnect import *
from .services.SimFrameScout import SimFrameScout
from datetime import datetime
import time
import mss
import numpy as np
import gc


class AircraftPositioningAgent:
    """
    Controls aircraft positioning in Microsoft Flight Simulator via SimConnect.
    
    This class provides methods to:
    - Position the aircraft at specific geographic coordinates
    - Set aircraft orientation (pitch, yaw, roll)
    - Control simulation clock
    - Capture screenshots of the simulation
    
    The agent uses SimConnect to communicate with MSFS, and mss/PIL for
    high-performance screen capture.
    
    Attributes:
        sm: SimConnect connection object
        aq: AircraftRequests for reading/writing aircraft data
        ae: AircraftEvents for triggering in-sim events
        simFrameDimensions: SimFrameScout for window dimension detection
    """

    def __init__(self, sim):
        """
        Initialize the AircraftPositioningAgent.
        
        Args:
            sim: SimConnect instance for MSFS communication
        """
        # Create SimConnect link
        self.sm = sim
        self.aq = AircraftRequests(self.sm, _time=500)
        self.ae = AircraftEvents(self.sm)

        self.simFrameDimensions = SimFrameScout()

        # Define events to trigger in MSFS
        # Freeze events prevent the aircraft from moving during capture
        self.event_freeze_altitude = self.ae.find("FREEZE_ALTITUDE_TOGGLE")
        self.event_freeze_lat_long = self.ae.find("FREEZE_LATITUDE_LONGITUDE_TOGGLE")
        self.event_pause = self.ae.find("PAUSE_ON")

        # Clock events for setting simulation time
        self.event_clock_hours_set = self.ae.find("CLOCK_HOURS_SET")
        self.event_clock_minutes_set = self.ae.find("CLOCK_MINUTES_SET")
        self.event_clock_seconds_set = self.ae.find("CLOCK_SECONDS_SET")

    def positionAircraftInSimAndTakeScreenshot(self, latitude, longitude, altitude, pitch, heading, roll, 
                                              screenshot_path, window_width, window_height, 
                                              setSimHour, setSimMin, excludeImg):
        """
        Positions the aircraft in MSFS and optionally captures a screenshot.
        
        This method performs the following steps:
        1. Convert angles to radians for SimConnect
        2. Set aircraft position and orientation
        3. Set simulation time
        4. Freeze the aircraft (prevent movement)
        5. Pause the simulation
        6. Capture screenshot (if excludeImg is False)
        
        Args:
            latitude: Aircraft latitude in degrees
            longitude: Aircraft longitude in degrees
            altitude: Aircraft altitude in feet
            pitch: Aircraft pitch angle in degrees (nose up/down)
            heading: Aircraft heading in degrees (compass direction)
            roll: Aircraft bank/roll angle in degrees (wing tilt)
            screenshot_path: Directory path for saving screenshots
            window_width: Width of screenshot in pixels
            window_height: Height of screenshot in pixels
            setSimHour: Simulation hour (0-23)
            setSimMin: Simulation minute (0-59)
            excludeImg: If True, skip screenshot creation (return timestamp only)
            
        Returns:
            str: Timestamp string used for the screenshot filename
        """
        try:
            # Convert degrees to radians (SimConnect uses radians)
            pitch = np.radians(pitch)
            heading = np.radians(heading)
            roll = np.radians(roll)
            
            # Set aircraft position in MSFS
            self.aq.set("PLANE_LATITUDE", latitude)    # Degrees
            self.aq.set("PLANE_LONGITUDE", longitude)  # Degrees
            self.aq.set("PLANE_ALTITUDE", altitude)    # Feet
            
            # Set aircraft orientation
            self.aq.set("PLANE_PITCH_DEGREES", pitch)          # Radians
            self.aq.set("PLANE_HEADING_DEGREES_TRUE", heading) # Radians
            self.aq.set("PLANE_BANK_DEGREES", roll)            # Radians

            # Set simulation time
            self.event_clock_hours_set(setSimHour)
            self.event_clock_minutes_set(setSimMin)

            # Freeze aircraft position to prevent movement during capture
            self.event_freeze_altitude()
            self.event_freeze_lat_long()
            self.event_pause()

            # Skip screenshot creation if requested
            if excludeImg:
                now = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
                print(f"[INFO] excludeImg=True - no screenshot, using name {now}")
                return now

            # Get the Microsoft Flight Simulator window
            window = pygetwindow.getWindowsWithTitle('Microsoft Flight Simulator')
            if window:
                sim_window = window[0]
                sim_window.activate()

                # Get original window dimensions
                left, top, right, bottom = sim_window.left, sim_window.top, sim_window.right, sim_window.bottom
                original_width = right - left
                original_height = bottom - top

                # Use custom dimensions if provided, otherwise use original
                width = window_width if window_width is not None else original_width
                height = window_height if window_height is not None else original_height

                # Calculate new bounding box centered on original window
                center_x = left + (original_width // 2)
                center_y = top + (original_height // 2)

                new_left = center_x - (width // 2)
                new_top = center_y - (height // 2)
                new_right = new_left + width
                new_bottom = new_top + height
                
                # Wait for MSFS to render the new position
                #time.sleep(30)

                # Capture screenshot using mss (fast screen capture library)
                with mss.mss() as sct:
                    monitor = {"left": int(new_left), "top": int(new_top), 
                              "width": int(width), "height": int(height)}
                    screenshot = sct.grab(monitor)

                    # Save the screenshot as PNG
                    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                    img.save(f"{screenshot_path}\\{now}.png")
                    print(f"Screenshot saved: {screenshot_path}\\{now}.png")

            else:
                print("Error: Microsoft Flight Simulator window not found.")
                exit()
                
            return now
            
        except Exception as e:
            print(f"[ERROR] SimConnect Error: {e}")
        finally: 
            # Force garbage collection to free memory
            gc.collect()
        
