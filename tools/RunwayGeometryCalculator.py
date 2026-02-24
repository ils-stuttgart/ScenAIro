"""
RunwayGeometryCalculator - Airport Runway Geometry Calculations
==============================================================

This module provides the RunwayGeometryCalculator class which handles
all geometric calculations related to airport runways.

The calculator can:
    - Calculate runway corner positions in local Cartesian coordinates
    - Handle runway heading/orientation rotation
    - Create/save/load airport configurations
    - Support different runway heights at start and end points

Author: ScenAIro Team
"""

import math
import json


class RunwayGeometryCalculator:
    """
    Handles runway geometry calculations for the ScenAIro application.
    
    This class calculates the Cartesian coordinates of runway corners
    based on runway parameters like width, length, heading, and heights.
    It also supports saving and loading airport configurations from JSON.
    
    Attributes:
        name: Airport name
        icao_code: ICAO code (e.g., "EDDV" for Hannover)
        runway_name: Runway designation (e.g., "09L")
        runway_width: Width of the runway in meters
        runway_length: Length of the runway in meters
        runway_heading: Heading of the runway in degrees
        runway_center: Dict with latitude, longitude, altitude
        start_height: Height at runway start (threshold) in meters
        end_height: Height at runway end in meters
        runway_attributes: Additional runway attributes dict
    """

    def __init__(self, name, icao_code, runway_name, runway_width, runway_length, runway_heading, 
                 center_lat, center_long, center_alt, start_height, end_height, runway_attributes):
        """
        Initialize the RunwayGeometryCalculator with airport parameters.
        
        Args:
            name: Airport name (e.g., "Hannover Airport")
            icao_code: ICAO code (e.g., "EDDV")
            runway_name: Runway designation (e.g., "09L", "24")
            runway_width: Runway width in meters
            runway_length: Runway length in meters
            runway_heading: Runway heading in degrees
            center_lat: Center latitude in degrees
            center_long: Center longitude in degrees
            center_alt: Center altitude in meters
            start_height: Height at runway start/threshold in meters
            end_height: Height at runway end in meters
            runway_attributes: Additional attributes dictionary
        """
        self.name = name
        self.icao_code = icao_code
        self.runway_name = runway_name
        self.runway_width = float(runway_width)
        self.runway_length = float(runway_length)
        self.runway_heading = float(runway_heading)
        self.runway_center = {"latitude": center_lat, "longitude": center_long, "altitude": center_alt}
        self.start_height = start_height
        self.end_height = end_height
        self.runway_attributes = runway_attributes

    def calculateRunwayCorners(self):
        """
        Calculate the Cartesian coordinates of all four runway corners.
        
        The corners are calculated relative to the runway center, taking into
        account the runway heading for proper orientation. Heights are assigned
        based on start_height (threshold end) and end_height (opposite end).
        
        Returns:
            dict: Dictionary with keys 'top_left', 'top_right', 'bottom_left', 'bottom_right',
                  each containing a tuple of (x, y, z) coordinates
        """
        heading_rad = math.radians(self.runway_heading)
        runwayAltitude = self.runway_center["altitude"]
        half_length = self.runway_length / 2
        half_width = self.runway_width / 2
        start_height = self.start_height
        end_height = self.end_height
        
        # Calculate heights at each corner based on runway slope
        startCornerHeight = runwayAltitude + (start_height - runwayAltitude)
        endCornerHeight = runwayAltitude + (end_height - runwayAltitude)
        
        # Calculate corner positions, applying rotation for runway heading
        corners = {
            # "top" refers to the end with higher elevation (threshold)
            "top_left": (*self.alignCornersWithRunwayHeading(-half_length, half_width, heading_rad), endCornerHeight),
            "top_right": (*self.alignCornersWithRunwayHeading(-half_length, -half_width, heading_rad), endCornerHeight),
            # "bottom" refers to the starting end
            "bottom_left": (*self.alignCornersWithRunwayHeading(half_length, half_width, heading_rad), startCornerHeight),
            "bottom_right": (*self.alignCornersWithRunwayHeading(half_length, -half_width, heading_rad), startCornerHeight),
        }
        return corners

    def alignCornersWithRunwayHeading(self, x, y, angle_rad):
        """
        Rotate a point around the origin based on the runway heading.
        
        Uses a 2D rotation matrix to transform coordinates from the local
        runway reference frame to the world reference frame.
        
        Args:
            x: X coordinate in local runway frame
            y: Y coordinate in local runway frame
            angle_rad: Rotation angle in radians
            
        Returns:
            tuple: (x_rot, y_rot) - rotated coordinates rounded to 2 decimal places
        """
        x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        return round(x_rot, 2), round(y_rot, 2)

    def createAirport(self):
        """
        Create a dictionary representation of the airport configuration.
        
        Returns:
            dict: Airport configuration containing all parameters
        """
        return {
            "airport_name": self.name,
            "icao_code": self.icao_code,
            "runway": {
                "name": self.runway_name,
                "width": self.runway_width,
                "length": self.runway_length,
                "heading": self.runway_heading,
                "center_coordinates": self.runway_center,
            },
            "start_height": self.start_height,
            "end_height": self.end_height,
            "attributes": self.runway_attributes,
        }

    @classmethod
    def createAirportConfig(cls, data):
        """
        Create a RunwayGeometryCalculator instance from a dictionary.
        
        Args:
            data: Dictionary containing airport configuration
            
        Returns:
            RunwayGeometryCalculator: New instance with loaded parameters
        """
        print("Creating RunwayCalc from dict")
        runway = data["runway"]
        center_coords = runway["center_coordinates"]
        return cls(
            name=data["airport_name"],
            icao_code=data["icao_code"],
            runway_name=runway["name"],
            runway_width=runway["width"],
            runway_length=runway["length"],
            runway_heading=runway["heading"],
            center_lat=center_coords["latitude"],
            center_long=center_coords["longitude"],
            center_alt=center_coords["altitude"],
            start_height=runway["start_height"],
            end_height=runway["end_height"],
            runway_attributes=runway.get("attributes", {})
        )

    def saveAirport(self, filename):
        """
        Save the airport configuration to a JSON file.
        
        Args:
            filename: Path to the output JSON file
        """
        print(f"Saving RunwayCalc to file: {filename}")
        with open(filename, "w") as file:
            json.dump(self.createAirport(), file, indent=4)

    @classmethod
    def loadAirport(cls, filename):
        """
        Load an airport configuration from a JSON file.
        
        Args:
            filename: Path to the input JSON file
            
        Returns:
            RunwayGeometryCalculator: New instance with loaded parameters
        """
        print(f"Loading RunwayCalc from file: {filename}")
        with open(filename, "r") as file:
            data = json.load(file)
        return cls.createAirportConfig(data)
