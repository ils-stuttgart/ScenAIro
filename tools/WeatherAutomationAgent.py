"""
WeatherAutomationAgent - MSFS Weather Control Automation
=======================================================

This module provides the WeatherAutomationAgent class which automates
the setting of weather conditions in Microsoft Flight Simulator by 
simulating mouse interactions.

The automation sequence is:
    1. Hover over the top area to reveal the menu bar
    2. Click on the weather icon in the menu
    3. Click on the weather preset dropdown
    4. Select the desired weather option
    5. Close the menu

Note: These coordinates are specific to your screen resolution and UI scaling.
You may need to adjust them for different display configurations.

Dependencies:
    - pyautogui: For mouse automation
    - pygetwindow: For window detection

Author: ScenAIro Team
"""

import time
import pyautogui
import pygetwindow as gw


class WeatherAutomationAgent:
    """
    Automates weather setting in Microsoft Flight Simulator.
    
    This class uses mouse automation to interact with the MSFS UI and set
    weather conditions. It simulates the sequence of clicks needed to
    navigate to and select weather presets.
    
    Attributes:
        window_title: Title of the MSFS window to interact with
        coords: Dictionary of screen coordinates for UI elements
    """

    def __init__(self, window_title="Microsoft Flight Simulator"):
        """
        Initialize the WeatherAutomationAgent.
        
        Args:
            window_title: The window title to search for (default: "Microsoft Flight Simulator")
        """
        self.window_title = window_title

        # ============================================================
        # COORDINATE CONFIGURATION (Relative to window)
        # ============================================================
        # IMPORTANT: These values must be adjusted to match your screen resolution and UI scaling!
        # Values are (X, Y) pixels relative to the top-left corner of the simulator window.
        
        self.coords = {
            # 1. Hover area (top center) to reveal the menu bar
            "hover_trigger": (1280, 45), 
            
            # 2. Weather icon in the menu bar
            "menu_icon": (1280, 120),    
            
            # 3. Weather preset dropdown in the opened window
            "dropdown_box": (950, 680),  
            
            # 4. Positions of individual options in the dropdown menu
            # (These depend on where the dropdown appears)
            "options": {
                "Clear Skies":(950, 780), 
                "Few Clouds":(950, 830), 
                "Scattered Clouds":(950, 875), 
                "Broken Clouds":(950, 925), 
                "High Level Clouds":(950, 975),
                "Overcast":(950, 1025), 
                "Rain":(950, 1075), 
                "Snow":(950, 1130), 
                "Light Thunderstorm":(950, 1180)
            },
            
            # 5. Close menu button
            "close_menu": (1720, 255),
        }
        
        # PyAutoGUI safety settings
        pyautogui.PAUSE = 0.5   # Short pause after each action
        pyautogui.FAILSAFE = True  # Moving mouse to corner aborts the script

    def set_weather(self, weather_condition):
        """
        Executes the click sequence to set the weather condition.
        
        Args:
            weather_condition: String name of the weather preset to set
                             (e.g., "Clear Skies", "Rain", "Snow")
                             
        Returns:
            bool: True if weather was set successfully, False otherwise
        """
        print(f"[WeatherAgent] Attempting to set weather to '{weather_condition}'...")

        window = self._get_window()
        if not window:
            print(f"[WeatherAgent] Error: Window '{self.window_title}' not found!")
            return False

        try:
            # Activate the window (bring to foreground)
            if not window.isActive:
                window.activate()
                time.sleep(1.0)  # Wait for window to become active

            # Calculate absolute window coordinates
            left, top = window.left, window.top

            # STEP 1: Hover to show menu
            self._perform_action(left, top, "hover_trigger", action="hover")
            time.sleep(1.0)  # Wait for UI animation to complete

            # STEP 2: Click menu icon
            self._perform_action(left, top, "menu_icon", action="hover")
            self._perform_action(left, top, "menu_icon", action="click")
            time.sleep(1.0)  # Wait for weather window to open

            # STEP 3: Open dropdown
            self._perform_action(left, top, "dropdown_box", action="click")
            time.sleep(0.5)  # Wait for dropdown to appear

            # STEP 4: Click weather option using _perform_action
            if weather_condition in self.coords["options"]:
                # Add the weather option coordinates to the coords dictionary temporarily
                opt_x, opt_y = self.coords["options"][weather_condition]
                self.coords["weather_option"] = (opt_x, opt_y)
                
                # Use perform_action to click the weather option
                self._perform_action(left, top, "weather_option", action="click")
                            
                # STEP 5: Close menu
                self._perform_action(left, top, "close_menu", action="click")
                time.sleep(0.5)
                
                return True
            else:
                print(f"[WeatherAgent] Error: Unknown weather option '{weather_condition}'")
                return False

        except Exception as e:
            print(f"[WeatherAgent] Critical error during click sequence: {e}")
            return False

    def _get_window(self):
        """
        Find and return the MSFS window.
        
        Returns:
            Window object if found, None otherwise
        """
        windows = gw.getWindowsWithTitle(self.window_title)
        if windows:
            return windows[0]
        return None

    def _perform_action(self, win_left, win_top, key, action="click"):
        """
        Perform a mouse action at a specified coordinate.
        
        Args:
            win_left: Window left coordinate (for absolute positioning)
            win_top: Window top coordinate (for absolute positioning)
            key: Name of the coordinate key in self.coords
            action: Type of action ("hover" or "click")
        """
        if key not in self.coords:
            return
        
        rel_x, rel_y = self.coords[key]
        abs_x = win_left + rel_x
        abs_y = win_top + rel_y
        
        # Add small random delay to appear more human-like
        import random
        time.sleep(random.uniform(0.1, 0.3))
        
        # Move mouse to position with slight variation for human-like movement
        pyautogui.moveTo(abs_x, abs_y, duration=random.uniform(0.3, 0.6))
        
        if action == "hover":
            # Simply hover briefly
            time.sleep(random.uniform(0.3, 0.5)) 
        else:
            # Simulate click with realistic delay (important for games!)
            print(f"[Click] Mouse press at {abs_x}, {abs_y}")
            # Human-like click: press and release with slight delay
            pyautogui.mouseDown(x=abs_x, y=abs_y)
            time.sleep(random.uniform(0.15, 0.25))  # Hold slightly variable
            pyautogui.mouseUp(x=abs_x, y=abs_y)
            time.sleep(random.uniform(0.4, 0.7))  # Wait for game to respond
