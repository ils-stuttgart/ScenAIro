"""
SettingsManager - Application Settings Management
==================================================

This module provides a centralized settings management system for the ScenAIro application.
It handles loading, saving, and providing default values for all configurable parameters.

The settings are organized into logical categories:
    - Window: Main window dimensions
    - Paths: File saving locations
    - Screen: Screenshot resolution settings
    - Camera: Camera field of view settings
    - UI: User interface layout settings

Settings are persisted to a JSON file in the config directory.

Author: ScenAIro Team
"""

import json
import os
from typing import Dict, Any


class SettingsManager:
    """
    Centralized settings management class.
    
    This class follows the singleton pattern to ensure only one instance
    manages all application settings. It provides:
        - Default values for all configurable parameters
        - Load/Save functionality to JSON
        - Type-safe access to settings
    
    Attributes:
        _instance: Singleton instance
        _initialized: Flag to track initialization
        settings: Dictionary containing all current settings
        settings_file: Path to the settings JSON file
    """
    
    # Singleton instance
    _instance = None
    _initialized = False
    
    # =========================================================================
    # DEFAULT SETTINGS
    # =========================================================================
    
    DEFAULT_SETTINGS = {
        # ----------------------------------------------------------------
        # Window Settings
        # ----------------------------------------------------------------
        # Main application window dimensions
        "window": {
            "width": 1400,          # Main window width in pixels
            "height": 820,          # Main window height in pixels
            "bg_color": "#f0f4f8"   # Background color (light grey-blue)
        },
        
        # ----------------------------------------------------------------
        # Path Settings
        # ----------------------------------------------------------------
        # File system paths for data export
        "paths": {
            "screenshot_path": r"C:\Users\mfs2024\Desktop\Saymon\Test",
            "config_path": "config"  # Relative path to config directory
        },
        
        # ----------------------------------------------------------------
        # Screen/Capture Settings
        # ----------------------------------------------------------------
        # Screenshot resolution and capture parameters
        "screen": {
            "width": 2560,          # Screenshot width in pixels
            "height": 1440          # Screenshot height in pixels
        },
        
        # ----------------------------------------------------------------
        # Camera Settings
        # ----------------------------------------------------------------
        # Field of view and camera parameters
        "camera": {
            "vertical_fov_radians": 0.8  # Vertical FOV in radians (~45.8 degrees)
        },
        
        # ----------------------------------------------------------------
        # UI Layout Settings
        # ----------------------------------------------------------------
        # Sidebar and panel dimensions
        "ui_layout": {
            "left_sidebar_width": 370,   # Left configuration panel width
            "right_sidebar_width": 250,  # Right legend panel width
            "plot_figsize": [6, 4],      # 3D plot figure size [width, height]
            "dist_figsize": [4, 2]       # 2D distribution plot size [width, height]
        },
        
        # ----------------------------------------------------------------
        # Plot Settings
        # ----------------------------------------------------------------
        # Visualization parameters
        "plot": {
            "point_size": 1,             # Size of scatter points
            "point_alpha": 0.5,          # Point transparency (0-1)
            "runway_alpha": 0.5,         # Runway polygon transparency
            "apex_point_size": 50        # Apex marker size
        }
    }
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __new__(cls):
        """
        Singleton pattern implementation.
        
        Ensures only one instance of SettingsManager exists throughout
        the application lifecycle.
        
        Returns:
            SettingsManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(SettingsManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize the settings manager.
        
        Loads settings from file if available, otherwise uses defaults.
        The initialization only occurs once due to the singleton pattern.
        """
        # Skip if already initialized
        if SettingsManager._initialized:
            return
        
        # Set up the settings file path
        # Located in the config directory relative to the project root
        self.settings_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "config",
            "settings.json"
        )
        
        # Load settings from file or use defaults
        self.settings = self._load_settings()
        
        # Mark as initialized
        SettingsManager._initialized = True
    
    # =========================================================================
    # LOAD/SAVE METHODS
    # =========================================================================
    
    def _load_settings(self) -> Dict[str, Any]:
        """
        Load settings from the JSON file.
        
        Attempts to load settings from the configured file path.
        If the file doesn't exist or is corrupted, returns default settings.
        
        Returns:
            Dict[str, Any]: Dictionary containing all settings
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                # This handles cases where new settings are added
                return self._merge_with_defaults(loaded_settings)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[SettingsManager] Error loading settings: {e}")
            print("[SettingsManager] Using default settings")
        
        # Return a deep copy of defaults
        return self._get_default_settings()
    
    def _merge_with_defaults(self, loaded: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge loaded settings with defaults to ensure completeness.
        
        This method recursively merges the loaded settings with the default
        settings, ensuring that any missing keys are filled with defaults.
        
        Args:
            loaded: Settings loaded from file
            
        Returns:
            Dict[str, Any]: Complete settings dictionary
        """
        result = self._get_default_settings()
        
        for category, values in loaded.items():
            if category in result and isinstance(values, dict):
                result[category].update(values)
            else:
                result[category] = values
        
        return result
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """
        Get a deep copy of the default settings.
        
        Returns:
            Dict[str, Any]: Copy of default settings
        """
        # Create a deep copy to avoid modifying the original
        import copy
        return copy.deepcopy(self.DEFAULT_SETTINGS)
    
    def save_settings(self) -> bool:
        """
        Save current settings to the JSON file.
        
        Persists all current settings to the configured file path.
        Creates the config directory if it doesn't exist.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Ensure config directory exists
            config_dir = os.path.dirname(self.settings_file)
            os.makedirs(config_dir, exist_ok=True)
            
            # Write settings to file with pretty formatting
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4, ensure_ascii=False)
            
            print(f"[SettingsManager] Settings saved to {self.settings_file}")
            return True
            
        except IOError as e:
            print(f"[SettingsManager] Error saving settings: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """
        Reset all settings to their default values.
        
        This method overwrites all current settings with the default
        values defined in DEFAULT_SETTINGS.
        """
        self.settings = self._get_default_settings()
        print("[SettingsManager] Settings reset to defaults")
    
    # =========================================================================
    # GETTER METHODS
    # =========================================================================
    
    def get(self, category: str, key: str = None) -> Any:
        """
        Get a setting value by category and optional key.
        
        This is the primary method for accessing settings values.
        
        Args:
            category: The settings category (e.g., 'window', 'paths')
            key: The specific setting key (optional, returns whole category if None)
            
        Returns:
            Any: The requested setting value
            
        Raises:
            KeyError: If the category or key doesn't exist
        """
        if category not in self.settings:
            raise KeyError(f"Settings category '{category}' not found")
        
        if key is None:
            return self.settings[category]
        
        if key not in self.settings[category]:
            raise KeyError(f"Setting '{key}' not found in category '{category}'")
        
        return self.settings[category][key]
    
    def set(self, category: str, key: str, value: Any) -> None:
        """
        Set a setting value by category and key.
        
        Updates a specific setting value. Changes are not persisted
        until save_settings() is called.
        
        Args:
            category: The settings category
            key: The specific setting key
            value: The new value to set
            
        Raises:
            KeyError: If the category doesn't exist
        """
        if category not in self.settings:
            raise KeyError(f"Settings category '{category}' not found")
        
        self.settings[category][key] = value
    
    # =========================================================================
    # CONVENIENCE PROPERTIES
    # =========================================================================
    
    @property
    def window_width(self) -> int:
        """Get the main window width."""
        return self.get("window", "width")
    
    @property
    def window_height(self) -> int:
        """Get the main window height."""
        return self.get("window", "height")
    
    @property
    def screenshot_path(self) -> str:
        """Get the screenshot save path."""
        return self.get("paths", "screenshot_path")
    
    @screenshot_path.setter
    def screenshot_path(self, value: str) -> None:
        """Set the screenshot save path."""
        self.set("paths", "screenshot_path", value)
    
    @property
    def screen_width(self) -> int:
        """Get the screenshot width."""
        return self.get("screen", "width")
    
    @property
    def screen_height(self) -> int:
        """Get the screenshot height."""
        return self.get("screen", "height")
    
    @property
    def vertical_fov_radians(self) -> float:
        """Get the vertical field of view in radians."""
        return self.get("camera", "vertical_fov_radians")
    
    @property
    def left_sidebar_width(self) -> int:
        """Get the left sidebar width."""
        return self.get("ui_layout", "left_sidebar_width")
    
    @property
    def right_sidebar_width(self) -> int:
        """Get the right sidebar width."""
        return self.get("ui_layout", "right_sidebar_width")
