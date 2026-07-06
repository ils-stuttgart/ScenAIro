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
        self.sim_window = None

        # Define events to trigger in MSFS
        # Freeze events prevent the aircraft from moving during capture
        self.event_freeze_altitude = self.ae.find("FREEZE_ALTITUDE_TOGGLE")
        self.event_freeze_lat_long = self.ae.find("FREEZE_LATITUDE_LONGITUDE_TOGGLE")
        self.event_pause = self.ae.find("PAUSE_ON")

        # Clock events for setting simulation time
        self.event_clock_hours_set = self.ae.find("CLOCK_HOURS_SET")
        self.event_clock_minutes_set = self.ae.find("CLOCK_MINUTES_SET")
        self.event_clock_seconds_set = self.ae.find("CLOCK_SECONDS_SET")
        
    def _get_sim_window(self):
        """Findet das MSFS-Fenster oder nutzt die bereits gespeicherte Referenz."""
        # Wenn wir das Fenster schon haben, prüfe ob es noch gültig ist
        if self.sim_window is not None:
            try:
                # Ein simpler Test-Zugriff auf ein Attribut, um zu prüfen, 
                # ob das Windows-Handle noch existiert.
                _ = self.sim_window.title
                return self.sim_window
            except Exception:
                # Das Fenster-Handle ist ungültig geworden (z.B. Sim gecrasht)
                self.sim_window = None 

        # Fenster neu suchen, falls wir es noch nicht haben oder es weg ist
        windows = pygetwindow.getWindowsWithTitle('Microsoft Flight Simulator')
        if windows:
            self.sim_window = windows[0]
            return self.sim_window
            
        return None

    def positionAircraftInSimAndTakeScreenshot(self, latitude, longitude, altitude, pitch, heading, roll, 
                                              screenshot_path, window_width, window_height, 
                                              setSimHour, setSimMin, excludeImg, custom_filename=None,
                                              pre_screenshot_delay=0.0):
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
             pre_screenshot_delay: Seconds to wait after freezing aircraft before
                 capturing screenshot. Used to allow scenery/runway textures to stream
                 in. Default: 0.0.
             
         Returns:
             str: Timestamp string used for the screenshot filename
        """
        try:
            # Freeze aircraft position to prevent movement during capture
            self.event_freeze_altitude()
            self.event_freeze_lat_long()
            self.event_pause()
            
            # Short pause for SDK to set Pause/Freeze safely
            time.sleep(0.05)
            
            # Convert degrees to radians (SimConnect uses radians)
            pitch = np.radians(pitch)
            heading = np.radians(heading)
            roll = np.radians(roll)
            
            # Set aircraft position in MSFS
            self.aq.set("PLANE_LATITUDE", latitude)    # Degrees
            self.aq.set("PLANE_LONGITUDE", longitude)  # Degrees
            self.aq.set("PLANE_ALTITUDE", altitude)    # Feet
            time.sleep(0.05)
            
            # Set aircraft orientation
            self.aq.set("PLANE_PITCH_DEGREES", pitch)          # Radians
            self.aq.set("PLANE_HEADING_DEGREES_TRUE", heading) # Radians
            self.aq.set("PLANE_BANK_DEGREES", roll)            # Radians
            time.sleep(0.05)

            # Set simulation time
            self.event_clock_hours_set(setSimHour)
            self.event_clock_minutes_set(setSimMin)

            # Skip screenshot creation if requested
            if excludeImg:
                now = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
                print(f"[INFO] excludeImg=True - no screenshot, using name {now}")
                time.sleep(0.1)
                return now

            # Adaptive pre-screenshot delay (allow graphics/runway textures to load)
            if pre_screenshot_delay > 0:
                print(f"[AircraftPositioningAgent] Waiting {pre_screenshot_delay}s before screenshot...")
                time.sleep(pre_screenshot_delay)
            else:
                time.sleep(0.2)

            # Get the Microsoft Flight Simulator window
            sim_window = self._get_sim_window()
            
            if sim_window:
                try:
                    sim_window.activate()
                except Exception as e:
                    # Manchmal blockiert Windows das "activate", aber wir können
                    # mit 'mss' trotzdem den Bildschirmbereich aufnehmen!
                    print(f"[WARN] Could not find window: {e}")

                # Get original window dimensions
                left, top, right, bottom = sim_window.left, sim_window.top, sim_window.right, sim_window.bottom
                original_width = right - left
                original_height = bottom - top

                # Use custom dimensions if provided, otherwise use original
                width = window_width if window_width is not None else original_width
                height = window_height if window_height is not None else original_height

                center_x = left + (original_width // 2)
                center_y = top + (original_height // 2)

                new_left = center_x - (width // 2)
                new_top = center_y - (height // 2)
                
                with mss.mss() as sct:
                    monitor = {"left": int(new_left), "top": int(new_top), 
                              "width": int(width), "height": int(height)}
                    screenshot = sct.grab(monitor)

                    if custom_filename:
                        file_base_name = custom_filename
                    else:
                        file_base_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    
                    img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                    img.save(f"{screenshot_path}\\{file_base_name}.png")
                    img.close()
                    print(f"Screenshot saved: {screenshot_path}\\{file_base_name}.png")

            else:
                print("[ERROR] Microsoft Flight Simulator window not found.")
                raise RuntimeError("MSFS window lost.")
                
            return file_base_name
            
        except Exception as e:
            print(f"[ERROR] SimConnect Error: {e}")
        finally: 
            # Force garbage collection to free memory
            gc.collect()
        
