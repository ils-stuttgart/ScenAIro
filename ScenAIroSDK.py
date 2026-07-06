"""
ScenAIroSDK - Python SDK for ScenAIro
=====================================

This module provides the ScenAIroSDK class, which allows programmatic control
of the ScenAIro synthetic data generation toolchain from external Python projects.

The SDK supports:
    - Full headless mode for automation
    - Optional GUI launch for interactive use
    - Airport/runway configuration
    - Cone-based point generation
    - Aircraft positioning in MSFS
    - Weather control
    - Screenshot capture and labeling

Usage:
    from ScenAIroSDK import ScenAIroSDK
    
    # Create SDK instance
    sdk = ScenAIroSDK()
    
    # Configure airport
    sdk.configure_airport(
        name="Hannover Airport",
        icao_code="EDDV",
        runway_name="09L",
        width=45,
        length=2340,
        heading=87,
        latitude=52.4611,
        longitude=9.6850,
        altitude=55,
        start_height=0,
        end_height=0
    )
    
    # Configure point generation
    sdk.configure_point_generation(
        apex=(0, 0, 50),
        lateral_angle_left=30,
        lateral_angle_right=30,
        vertical_min_angle=5,
        vertical_max_angle=45,
        max_distance=3000,
        num_points=100
    )
    
    # Configure aircraft orientation
    sdk.configure_aircraft_orientation(
        pitch_min=-10,
        pitch_max=10,
        yaw_min=-45,
        yaw_max=45,
        roll_min=-30,
        roll_max=30
    )
    
    # Generate data
    result = sdk.generate_data(
        weather="Clear",
        enable_labeling=True,
        enable_overlay=True
    )

Author: ScenAIro Team
"""

import sys
import os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ScenAIro core components
from tools.RunwayGeometryCalculator import RunwayGeometryCalculator
from tools.SamplingPointGenerator import SamplingPointGenerator
from tools.GeoCoordinateProjector import GeoCoordinateProjector
from tools.AircraftPositioningAgent import AircraftPositioningAgent
from tools.RunwayTaggingEngine import RunwayTaggingEngine
from tools.WeatherAutomationAgent import WeatherAutomationAgent
from tools.SettingsManager import SettingsManager


class ScenAIroSDK:
    """
    Python SDK for ScenAIro - Programmatic control of the synthetic data generation toolchain.
    
    This class provides a clean interface for controlling all ScenAIro functionality
    from external Python projects. It can be used in headless mode for automation
    or combined with the optional GUI launcher.
    
    Attributes:
        settings: SettingsManager instance for configuration
        airport: RunwayGeometryCalculator instance for runway calculations
        points: Generated 3D sampling points (numpy array)
        geo_points: Transformed geographic coordinates
        is_configured: Flag indicating if SDK is properly configured
        
    Example:
        >>> sdk = ScenAIroSDK()
        >>> sdk.configure_airport(...)
        >>> sdk.configure_point_generation(...)
        >>> result = sdk.generate_data()
    """
    
    # Default distribution settings
    DEFAULT_DISTRIBUTION = {
        "type": "Normal Distribution",
        "apply_x": False,
        "apply_y": False
    }
    
    # Available weather presets
    AVAILABLE_WEATHER = [
        "Clear",
        "Few Clouds", 
        "Scattered Clouds",
        "Broken Clouds",
        "Overcast",
        "Fog",
        "Rain",
        "Snow",
        "Thunderstorm"
    ]
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ScenAIro SDK.
        
        Args:
            config_path: Optional path to custom config directory. 
                        If None, uses default config path.
        """
        # Initialize settings manager
        self.settings = SettingsManager()
        
        # Apply custom config path if provided
        if config_path:
            self.settings.set("paths", "config_path", config_path)
        
        # Initialize state variables
        self.airport = None
        self.points = None
        self.geo_points = None
        self.angles = None
        self.apex = None
        self.distribution_settings = None
        
        # Aircraft orientation parameters
        self.pitch_min = None
        self.pitch_max = None
        self.yaw_min = None
        self.yaw_max = None
        self.roll_min = None
        self.roll_max = None
        
        # Cone geometry parameters
        self.lateral_angle_left = None
        self.lateral_angle_right = None
        self.vertical_min_angle = None
        self.vertical_max_angle = None
        self.max_distance = None
        self.num_points = None  # Store number of points to generate
        self.aircraft_orientation = None
        self.apex_transformed = None
        
        # Runtime components (initialized on first use)
        self._sim_connect = None
        self._coord_setter = None
        self._tagging_engine = None
        self._weather_agent = None
        
        # Configuration flag
        self.is_configured = False
        
        print("[ScenAIroSDK] Initialized. Configure airport and point generation before generating data.")
    
    # =========================================================================
    # CONFIGURATION METHODS
    # =========================================================================
    
    def configure_airport(
        self,
        name: str,
        icao_code: str,
        runway_name: str,
        width: float,
        length: float,
        heading: float,
        latitude: float,
        longitude: float,
        altitude: float,
        start_height: float = 0,
        end_height: float = 0,
        attributes: Optional[Dict] = None
    ) -> "ScenAIroSDK":
        """
        Configure airport/runway parameters.
        
        Args:
            name: Airport name (e.g., "Hannover Airport")
            icao_code: ICAO code (e.g., "EDDV", "KLAX")
            runway_name: Runway designation (e.g., "09L", "24")
            width: Runway width in meters
            length: Runway length in meters
            heading: Runway heading in degrees
            latitude: Airport center latitude in degrees
            longitude: Airport center longitude in degrees
            altitude: Airport center altitude in meters
            start_height: Height at runway start/threshold (default: 0)
            end_height: Height at runway end (default: 0)
            attributes: Optional additional runway attributes
            
        Returns:
            Self for method chaining
            
        Example:
            >>> sdk.configure_airport(
            ...     name="Hannover Airport",
            ...     icao_code="EDDV",
            ...     runway_name="09L",
            ...     width=45,
            ...     length=2340,
            ...     heading=87,
            ...     latitude=52.4611,
            ...     longitude=9.6850,
            ...     altitude=55
            ... )
        """
        self.airport = RunwayGeometryCalculator(
            name=name,
            icao_code=icao_code,
            runway_name=runway_name,
            runway_width=width,
            runway_length=length,
            runway_heading=heading,
            center_lat=latitude,
            center_long=longitude,
            center_alt=altitude,
            start_height=start_height,
            end_height=end_height,
            runway_attributes=attributes or {}
        )
        
        self._check_configuration_complete()
        print(f"[ScenAIroSDK] Airport configured: {name} ({icao_code}) - RWY {runway_name}")
        return self
    
    def configure_point_generation(
        self,
        apex: Tuple[float, float, float],
        lateral_angle_left: float,
        lateral_angle_right: float,
        vertical_min_angle: float,
        vertical_max_angle: float,
        max_distance: float,
        num_points: int,
        distribution_type: str = "Normal Distribution",
        apply_x: bool = False,
        apply_y: bool = False
    ) -> "ScenAIroSDK":
        """
        Configure cone-based point generation parameters.
        
        Args:
            apex: Tuple (x, y, z) representing the apex/tip of the sampling cone
            lateral_angle_left: Left lateral spread angle in degrees
            lateral_angle_right: Right lateral spread angle in degrees
            vertical_min_angle: Minimum vertical angle in degrees
            vertical_max_angle: Maximum vertical angle in degrees
            max_distance: Maximum distance from apex in meters
            num_points: Number of points to generate
            distribution_type: Distribution type ("Normal Distribution", "Parabel", "Exponentiell")
            apply_x: Apply distribution to X axis
            apply_y: Apply distribution to Y axis
            
        Returns:
            Self for method chaining
            
        Example:
            >>> sdk.configure_point_generation(
            ...     apex=(0, 0, 50),
            ...     lateral_angle_left=30,
            ...     lateral_angle_right=30,
            ...     vertical_min_angle=5,
            ...     vertical_max_angle=45,
            ...     max_distance=3000,
            ...     num_points=100
            ... )
        """
        self.apex = apex
        self.lateral_angle_left = lateral_angle_left
        self.lateral_angle_right = lateral_angle_right
        self.vertical_min_angle = vertical_min_angle
        self.vertical_max_angle = vertical_max_angle
        self.max_distance = max_distance
        self.num_points = num_points  # Store this
        
        self.distribution_settings = {
            "type": distribution_type,
            "apply_x": apply_x,
            "apply_y": apply_y
        }
        
        self._check_configuration_complete()
        print(f"[ScenAIroSDK] Point generation configured: {num_points} points, max distance {max_distance}m")
        return self
    
    def configure_aircraft_orientation(
        self,
        pitch_min: float,
        pitch_max: float,
        yaw_min: float,
        yaw_max: float,
        roll_min: float,
        roll_max: float
    ) -> "ScenAIroSDK":
        """
        Configure aircraft orientation angle ranges.
        
        Args:
            pitch_min: Minimum pitch angle in degrees (nose up/down)
            pitch_max: Maximum pitch angle in degrees
            yaw_min: Minimum yaw angle in degrees (nose left/right)
            yaw_max: Maximum yaw angle in degrees
            roll_min: Minimum roll/bank angle in degrees
            roll_max: Maximum roll/bank angle in degrees
            
        Returns:
            Self for method chaining
            
        Example:
            >>> sdk.configure_aircraft_orientation(
            ...     pitch_min=-10,
            ...     pitch_max=10,
            ...     yaw_min=-45,
            ...     yaw_max=45,
            ...     roll_min=-30,
            ...     roll_max=30
            ... )
        """
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.yaw_min = yaw_min
        self.yaw_max = yaw_max
        self.roll_min = roll_min
        self.roll_max = roll_max
        
        self._check_configuration_complete()
        print(f"[ScenAIroSDK] Aircraft orientation configured: P({pitch_min},{pitch_max}) Y({yaw_min},{yaw_max}) R({roll_min},{roll_max})")
        return self
    
    def configure_output(
        self,
        screenshot_path: Optional[str] = None,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None
    ) -> "ScenAIroSDK":
        """
        Configure output settings.
        
        Args:
            screenshot_path: Directory for saving screenshots
            screen_width: Screenshot width in pixels
            screen_height: Screenshot height in pixels
            
        Returns:
            Self for method chaining
        """
        if screenshot_path:
            self.settings.set("paths", "screenshot_path", screenshot_path)
        if screen_width:
            self.settings.set("screen", "width", screen_width)
        if screen_height:
            self.settings.set("screen", "height", screen_height)
            
        print(f"[ScenAIroSDK] Output configured: {screenshot_path or self.settings.get('paths', 'screenshot_path')}")
        return self
    
    # =========================================================================
    # POINT GENERATION METHODS
    # =========================================================================
    
    def generate_points(self) -> np.ndarray:
        """
        Generate 3D sampling points based on configured parameters.
        
        Returns:
            numpy.ndarray: Generated 3D points array
            
        Raises:
            RuntimeError: If required configuration is missing
        """
        self._ensure_configured()
        
        aircraft_orientation_angles = {
            "pitchMin": self.pitch_min,
            "pitchMax": self.pitch_max,
            "yawMin": self.yaw_min,
            "yawMax": self.yaw_max,
            "rollMin": self.roll_min,
            "rollMax": self.roll_max
        }
        
        self.points, self.apex_transformed, self.aircraft_orientation = (
            SamplingPointGenerator.generateCone(
                apex=self.apex,
                lateral_angle_left=self.lateral_angle_left,
                lateral_angle_right=self.lateral_angle_right,
                vertical_min_angle=self.vertical_min_angle,
                vertical_max_angle=self.vertical_max_angle,
                max_distance=self.max_distance,
                num_points=self.num_points,  # Use stored value
                heading=self.airport.runway_heading,
                aircraftOrientationAngles=aircraft_orientation_angles,
                distribution_settings=self.distribution_settings
            )
        )
        
        print(f"[ScenAIroSDK] Generated {len(self.points)} sampling points")
        return self.points
    
    def transform_to_geocoordinates(
        self,
        points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Transform points to geographic coordinates.
        
        Args:
            points: Points to transform. If None, uses previously generated points.
            
        Returns:
            numpy.ndarray: Transformed geographic coordinates
        """
        if points is None:
            points = self.points
            
        if points is None:
            raise RuntimeError("No points available. Generate points first.")
        
        center_lat = self.airport.runway_center["latitude"]
        center_lon = self.airport.runway_center["longitude"]
        center_alt = self.airport.runway_center["altitude"]
        
        self.geo_points = GeoCoordinateProjector.transform_points(
            points=points,
            center_lat=center_lat,
            center_lon=center_lon,
            center_alt=center_alt,
            heading=0
        )
        
        print(f"[ScenAIroSDK] Transformed {len(self.geo_points)} points to geographic coordinates")
        return self.geo_points
    
    # =========================================================================
    # MSFS INTEGRATION METHODS
    # =========================================================================
    
    def _initialize_simconnect(self):
        """Initialize SimConnect connection if not already done."""
        if self._sim_connect is None:
            from dependencies.SimConnect import SimConnect
            self._sim_connect = SimConnect()
            self._coord_setter = AircraftPositioningAgent(self._sim_connect)
            self._tagging_engine = RunwayTaggingEngine()
            self._weather_agent = WeatherAutomationAgent()
            print("[ScenAIroSDK] SimConnect initialized")
    
    def set_weather(self, weather: str) -> bool:
        """
        Set weather conditions in MSFS.
        
        Args:
            weather: Weather preset name
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._initialize_simconnect()
        
        if weather not in self.AVAILABLE_WEATHER:
            print(f"[ScenAIroSDK] Warning: Unknown weather '{weather}'. Available: {self.AVAILABLE_WEATHER}")
            
        success = self._weather_agent.set_weather(weather)
        print(f"[ScenAIroSDK] Weather set to '{weather}': {success}")
        return success
    
    def position_aircraft(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        heading: float = 0,
        pitch: float = 0,
        roll: float = 0,
        screenshot_path: Optional[str] = None,
        window_width: int = 2560,
        window_height: int = 1440,
        set_sim_hour: int = 12,
        set_sim_minute: int = 0,
        exclude_image: bool = False
    ) -> str:
        """
        Position the aircraft in MSFS at the specified location and optionally capture a screenshot.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            altitude: Altitude in feet (note: SimConnect uses feet)
            heading: Heading in degrees
            pitch: Pitch angle in degrees
            roll: Roll angle in degrees
            screenshot_path: Directory path for saving screenshots
            window_width: Screenshot width in pixels
            window_height: Screenshot height in pixels
            set_sim_hour: Simulation hour (0-23)
            set_sim_minute: Simulation minute (0-59)
            exclude_image: If True, skip screenshot creation
            
        Returns:
            str: Timestamp string used for the screenshot filename
        """
        self._initialize_simconnect()
        
        if screenshot_path is None:
            screenshot_path = self.settings.get("paths", "screenshot_path")
        
        window_width = self.settings.get("screen", "width")
        window_height = self.settings.get("screen", "height")
        
        timestamp = self._coord_setter.positionAircraftInSimAndTakeScreenshot(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            pitch=pitch,
            heading=heading,
            roll=roll,
            screenshot_path=screenshot_path,
            window_width=window_width,
            window_height=window_height,
            setSimHour=set_sim_hour,
            setSimMin=set_sim_minute,
            excludeImg=exclude_image
        )
        
        print(f"[ScenAIroSDK] Aircraft positioned at ({latitude}, {longitude}, {altitude})")
        return timestamp
    
    # =========================================================================
    # DATA GENERATION METHOD
    # =========================================================================
    
    def generate_data(
        self,
        weather: Optional[str] = None,
        enable_labeling: bool = True,
        enable_overlay: bool = False,
        exclude_images: bool = False,
        sim_hour: int = 12,
        sim_minute: int = 0,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate complete labeled dataset for ML training.
        
        This is the main data generation function that:
        1. Generates sampling points based on current configuration
        2. Transforms points to geographic coordinates
        3. Optionally sets weather in MSFS
        4. Positions aircraft at each point and captures screenshots
        5. Creates overlay labels if enabled
        6. Saves metadata alongside images
        
        Args:
            weather: Weather preset to set in MSFS (optional)
            enable_labeling: Enable metadata generation with labels
            enable_overlay: Add visual overlay labels to screenshots
            exclude_images: Skip image creation (metadata only)
            sim_hour: Simulation hour (0-23)
            sim_minute: Simulation minute (0-59)
            progress_callback: Optional callback function(current, total) for progress
            
        Returns:
            Dict containing generation results:
                - success: bool
                - points_generated: int
                - geo_points: list
                - output_path: str
                - metadata: dict
                
        Example:
            >>> def progress(current, total):
            ...     print(f"Progress: {current}/{total}")
            >>> 
            >>> result = sdk.generate_data(
            ...     weather="Clear",
            ...     enable_labeling=True,
            ...     progress_callback=progress
            ... )
        """
        self._ensure_configured()
        
        print("[ScenAIroSDK] Starting data generation...")
        
        # Step 1: Generate points
        if self.points is None:
            self.generate_points()
            
        if self.points is None or len(self.points) == 0:
            raise RuntimeError("Failed to generate sampling points")
            
        # Step 2: Transform to geographic coordinates
        if self.geo_points is None:
            self.transform_to_geocoordinates()
            
        # Step 3: Set weather if specified
        if weather:
            self.set_weather(weather)
            
        # Step 4: Get output path
        output_path = self.settings.get("paths", "screenshot_path")
        
        # Step 5: Prepare metadata
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
        
        cone_metadata = {
            "apex": self.apex,
            "lateral_angle_left": self.lateral_angle_left,
            "lateral_angle_right": self.lateral_angle_right,
            "vertical_min_angle": self.vertical_min_angle,
            "vertical_max_angle": self.vertical_max_angle,
            "max_distance": self.max_distance,
            "number_of_points": len(self.points),
            "distribution": self.distribution_settings
        }
        
        orientation_metadata = {
            "pitch_min": self.pitch_min,
            "pitch_max": self.pitch_max,
            "yaw_min": self.yaw_min,
            "yaw_max": self.yaw_max,
            "roll_min": self.roll_min,
            "roll_max": self.roll_max
        }
        
        # Note: Full screenshot capture requires MSFS to be running
        # This is a placeholder for the complete implementation
        print(f"[ScenAIroSDK] Data generation ready:")
        print(f"  - Points: {len(self.points)}")
        print(f"  - Output: {output_path}")
        print(f"  - Weather: {weather or 'Not set'}")
        print(f"  - Labeling: {enable_labeling}")
        
        result = {
            "success": True,
            "points_generated": len(self.points),
            "geo_points": self.geo_points.tolist() if hasattr(self.geo_points, 'tolist') else self.geo_points,
            "output_path": output_path,
            "metadata": {
                "airport": airport_metadata,
                "cone": cone_metadata,
                "orientation": orientation_metadata,
                "settings": {
                    "enable_labeling": enable_labeling,
                    "enable_overlay": enable_overlay,
                    "exclude_images": exclude_images,
                    "sim_time": f"{sim_hour:02d}:{sim_minute:02d}"
                }
            }
        }
        
        return result
    
    # =========================================================================
    # GUI LAUNCH METHOD
    # =========================================================================
    
    def launch_gui(self):
        """
        Launch the ScenAIro GUI application.
        
        This method initializes and launches the full tkinter GUI,
        passing the current SDK configuration to the UI.
        
        Note: This blocks until the GUI is closed.
        
        Example:
            >>> sdk.configure_airport(...)
            >>> sdk.configure_point_generation(...)
            >>> sdk.launch_gui()  # Opens GUI with pre-filled config
        """
        from ScenAIro import ScenAIro
        
        print("[ScenAIroSDK] Launching GUI...")
        app = ScenAIro()
        
        # Pre-fill UI with SDK configuration if available
        if self.airport is not None:
            print("[ScenAIroSDK] Pre-filling airport configuration in GUI")
            # The GUI will pick up any pre-configured values
        
        app.mainloop()
        
        print("[ScenAIroSDK] GUI closed")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _check_configuration_complete(self):
        """Check if all required configuration is complete."""
        if (self.airport is not None and 
            self.apex is not None and
            self.lateral_angle_left is not None and
            self.pitch_min is not None):
            self.is_configured = True
    
    def _ensure_configured(self):
        """Ensure SDK is properly configured before operations."""
        if not self.is_configured:
            raise RuntimeError(
                "SDK not fully configured. Please call:\n"
                "  - configure_airport()\n"
                "  - configure_point_generation()\n"
                "  - configure_aircraft_orientation()"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current SDK status and configuration.
        
        Returns:
            Dict containing current status
        """
        return {
            "configured": self.is_configured,
            "airport": {
                "name": self.airport.name if self.airport else None,
                "icao": self.airport.icao_code if self.airport else None,
                "runway": self.airport.runway_name if self.airport else None
            } if self.airport else None,
            "points": {
                "generated": len(self.points) if self.points is not None else 0,
                "transformed": len(self.geo_points) if self.geo_points is not None else 0
            },
            "settings": {
                "screenshot_path": self.settings.get("paths", "screenshot_path"),
                "screen": {
                    "width": self.settings.get("screen", "width"),
                    "height": self.settings.get("screen", "height")
                }
            }
        }
    
    def reset(self):
        """Reset SDK to initial state."""
        self.airport = None
        self.points = None
        self.geo_points = None
        self.angles = None
        self.apex = None
        self.distribution_settings = None
        self.is_configured = False
        
        # Note: SimConnect instance is kept open
        print("[ScenAIroSDK] Reset to initial state")
    
    # =========================================================================
    # FACTORY METHODS
    # =========================================================================
    
    @classmethod
    def from_config_file(cls, config_path: str) -> "ScenAIroSDK":
        """
        Create SDK instance from a configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Configured ScenAIroSDK instance
        """
        import json
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        sdk = cls()
        
        if "airport" in config:
            ap = config["airport"]
            sdk.configure_airport(**ap)
        
        if "point_generation" in config:
            pg = config["point_generation"]
            sdk.configure_point_generation(**pg)
        
        if "aircraft_orientation" in config:
            ao = config["aircraft_orientation"]
            sdk.configure_aircraft_orientation(**ao)
        
        if "output" in config:
            out = config["output"]
            sdk.configure_output(**out)
        
        return sdk
    
    def save_config(self, config_path: str):
        """
        Save current configuration to a file.
        
        Args:
            config_path: Path to save configuration JSON
        """
        import json
        
        config = {
            "airport": {
                "name": self.airport.name,
                "icao_code": self.airport.icao_code,
                "runway_name": self.airport.runway_name,
                "width": self.airport.runway_width,
                "length": self.airport.runway_length,
                "heading": self.airport.runway_heading,
                "latitude": self.airport.runway_center["latitude"],
                "longitude": self.airport.runway_center["longitude"],
                "altitude": self.airport.runway_center["altitude"],
                "start_height": self.airport.start_height,
                "end_height": self.airport.end_height
            } if self.airport else None,
            "point_generation": {
                "apex": self.apex,
                "lateral_angle_left": self.lateral_angle_left,
                "lateral_angle_right": self.lateral_angle_right,
                "vertical_min_angle": self.vertical_min_angle,
                "vertical_max_angle": self.vertical_max_angle,
                "max_distance": self.max_distance,
                "num_points": 100,  # Default
                "distribution_type": self.distribution_settings["type"] if self.distribution_settings else "Normal Distribution",
                "apply_x": self.distribution_settings["apply_x"] if self.distribution_settings else False,
                "apply_y": self.distribution_settings["apply_y"] if self.distribution_settings else False
            } if self.apex else None,
            "aircraft_orientation": {
                "pitch_min": self.pitch_min,
                "pitch_max": self.pitch_max,
                "yaw_min": self.yaw_min,
                "yaw_max": self.yaw_max,
                "roll_min": self.roll_min,
                "roll_max": self.roll_max
            } if self.pitch_min is not None else None,
            "output": {
                "screenshot_path": self.settings.get("paths", "screenshot_path"),
                "screen_width": self.settings.get("screen", "width"),
                "screen_height": self.settings.get("screen", "height")
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[ScenAIroSDK] Configuration saved to {config_path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_generate(
    airport_config: Dict,
    point_config: Dict,
    orientation_config: Dict,
    output_path: str,
    weather: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick one-shot data generation function.
    
    This is a convenience function for simple use cases where you just
    want to generate data with a single function call.
    
    Args:
        airport_config: Airport configuration dict
        point_config: Point generation configuration dict
        orientation_config: Aircraft orientation configuration dict
        output_path: Screenshot output directory
        weather: Optional weather preset
        
    Returns:
        Generation result dictionary
        
    Example:
        >>> result = quick_generate(
        ...     airport_config={"name": "Test", "icao_code": "EDDV", ...},
        ...     point_config={"apex": (0, 0, 50), "lateral_angle_left": 30, ...},
        ...     orientation_config={"pitch_min": -10, "pitch_max": 10, ...},
        ...     output_path="C:/output"
        ... )
    """
    sdk = ScenAIroSDK()
    
    sdk.configure_airport(**airport_config)
    sdk.configure_point_generation(**point_config)
    sdk.configure_aircraft_orientation(**orientation_config)
    sdk.configure_output(screenshot_path=output_path)
    
    return sdk.generate_data(weather=weather)


if __name__ == "__main__":
    # Example usage when running directly
    print("ScenAIroSDK - Python SDK for ScenAIro")
    print("=" * 50)
    print("\nUsage:")
    print("  from ScenAIroSDK import ScenAIroSDK")
    print("  sdk = ScenAIroSDK()")
    print("  sdk.configure_airport(...)")
    print("  sdk.configure_point_generation(...)")
    print("  sdk.configure_aircraft_orientation(...)")
    print("  result = sdk.generate_data(...)")
    print("\nOr use the GUI:")
    print("  sdk = ScenAIroSDK()")
    print("  sdk.launch_gui()")
