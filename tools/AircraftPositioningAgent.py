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
    def __init__(self, sim):
        # Create SimConnect link
        self.sm = sim
        self.aq = AircraftRequests(self.sm, _time=500)
        self.ae = AircraftEvents(self.sm)

        self.simFrameDimensions = SimFrameScout()

        # Define events to trigger
        self.event_freeze_altitude = self.ae.find("FREEZE_ALTITUDE_TOGGLE")
        self.event_freeze_lat_long = self.ae.find("FREEZE_LATITUDE_LONGITUDE_TOGGLE")
        self.event_pause = self.ae.find("PAUSE_ON")

    def positionAircraftInSimAndTakeScreenshot(self, latitude, longitude, altitude, pitch, heading, roll, screenshot_path, window_width, window_height):
        try:
            heading = np.radians(heading)
            self.aq.set("PLANE_LATITUDE", latitude) #radian
            self.aq.set("PLANE_LONGITUDE", longitude) #radian
            self.aq.set("PLANE_ALTITUDE", altitude) #feet
            self.aq.set("PLANE_PITCH_DEGREES", pitch) #feet
            self.aq.set("PLANE_HEADING_DEGREES_TRUE", heading) #radians
            self.aq.set("PLANE_BANK_DEGREES", roll) #radians

            # Trigger freeze events
            self.event_freeze_altitude()
            self.event_freeze_lat_long()
            self.event_pause()

            # Get the window with the title 'Microsoft Flight Simulator'
            window = pygetwindow.getWindowsWithTitle('Microsoft Flight Simulator')
            if window:
                sim_window = window[0]
                sim_window.activate()

                # Originalgröße des Fensters ermitteln
                left, top, right, bottom = sim_window.left, sim_window.top, sim_window.right, sim_window.bottom
                original_width = right - left
                original_height = bottom - top

                # Falls eine benutzerdefinierte Größe angegeben ist, diese verwenden
                width = window_width if window_width is not None else original_width
                height = window_height if window_height is not None else original_height

                # Berechnung der neuen Bounding Box mit Zentrum des ursprünglichen Fensters
                center_x = left + (original_width // 2)
                center_y = top + (original_height // 2)

                new_left = center_x - (width // 2)
                new_top = center_y - (height // 2)
                new_right = new_left + width
                new_bottom = new_top + height

                # Take a screenshot of the specific window region using mss
                with mss.mss() as sct:
                    monitor = {"left": int(new_left), "top": int(new_top), "width": int(width), "height": int(height)}
                    screenshot = sct.grab(monitor)

                    # Save the screenshot
                    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                    img.save(f"{screenshot_path}\\{now}.png")
                    print(f"Screenshot saved: {screenshot_path}\\{now}.png")

            else:
                print("Fehler: Microsoft Flight Simulator Fenster nicht gefunden.")
                exit()
            return now
        except Exception as e:
            print(F"[ERROR] SimConncet-Error {e}")
        finally: 
            gc.collect()
        
