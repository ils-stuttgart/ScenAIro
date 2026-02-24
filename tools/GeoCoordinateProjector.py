"""
GeoCoordinateProjector - Geographic Coordinate Transformation
=============================================================

This module provides the GeoCoordinateProjector class which transforms
3D Cartesian coordinates (x, y, z) to geographic coordinates (latitude, longitude, altitude).

The transformation uses the WGS84 ellipsoid for accurate geodesic calculations,
which is the standard coordinate system used by GPS and Microsoft Flight Simulator.

Author: ScenAIro Team
"""

import numpy as np
from pyproj import Geod


class GeoCoordinateProjector:
    """
    Transforms 3D Cartesian coordinates to geographic coordinates.
    
    This class uses the pyproj library with WGS84 ellipsoid to perform
    accurate geodesic transformations. It calculates the new latitude and
    longitude based on distance and azimuth from a reference point.
    
    Attributes:
        geod: Geod object initialized with WGS84 ellipsoid parameters
    """
    
    # Initialize geodesic calculator with WGS84 ellipsoid (standard GPS coordinates)
    geod = Geod(ellps='WGS84')

    @staticmethod
    def transform_points(points, center_lat, center_lon, center_alt, heading):
        """
        Transform Cartesian points (x, y, z) to geographic coordinates (lat, lon, alt).
        
        This method converts local 3D coordinates to geographic coordinates by:
        1. Calculating the 2D distance from the origin
        2. Computing the azimuth (bearing) direction
        3. Using geodesic forward calculation to get lat/lon
        4. Adding the altitude offset
        
        Args:
            points: Array/List of (x, y, z) points in Cartesian coordinates.
                   x: Forward distance, y: Lateral offset, z: Vertical offset
            center_lat: Latitude of the reference point (center) in degrees
            center_lon: Longitude of the reference point (center) in degrees
            center_alt: Altitude of the reference point (center) in meters
            heading: Direction/heading in degrees (affects coordinate transformation)
            
        Returns:
            list: List of transformed points as (latitude, longitude, altitude) tuples
        """
        heading_rad = np.radians(heading)
        transformed_points = []

        for x, y, z in points:
            # Calculate 2D distance from origin (x, y plane)
            distance = np.sqrt(x ** 2 + y ** 2)
            
            # Calculate azimuth direction, adjusting for heading
            azimuth = np.degrees(heading_rad + np.arctan2(y, x))
            
            # Perform geodesic forward calculation to get new lat/lon
            lon_new, lat_new, back_azimuth = GeoCoordinateProjector.geod.fwd(
                center_lon, center_lat, azimuth, distance
            )
            
            # Add altitude offset to get final altitude
            alt_new = center_alt + z
            transformed_points.append((lat_new, lon_new, alt_new))

        return transformed_points
