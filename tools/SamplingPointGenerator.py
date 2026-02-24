"""
SamplingPointGenerator - 3D Point Cloud Generation for Training Data
====================================================================

This module provides the SamplingPointGenerator class which generates
3D point clouds using a cone-based sampling model.

The generator creates points within a cone volume defined by:
    - Apex (tip of the cone)
    - Lateral angles (left/right spread)
    - Vertical angles (up/down spread)
    - Maximum distance

The distribution of points can be customized using various probability
distributions (Uniform, Parabolic, Exponential).

Author: ScenAIro Team
"""

import numpy as np


class SamplingPointGenerator:
    """
    Generates 3D sampling points within a cone volume.
    
    This class creates point clouds for synthetic training data generation.
    Points are generated in a cone shape extending from an apex point,
    with configurable spread angles and distribution patterns.
    
    The coordinate system used:
        - X-axis: Forward direction (along runway heading)
        - Y-axis: Lateral direction (left/right)
        - Z-axis: Vertical direction (up/down)
    """

    def __init__(self):
        """Initialize the SamplingPointGenerator."""
        super().__init__()

    @staticmethod
    def __transformAimingPoint(apex, heading_rad):
        """
        Rotate the apex point according to the heading around the Z-axis.
        
        Args:
            apex: Tuple (x, y, z) representing apex coordinates
            heading_rad: Heading angle in radians
            
        Returns:
            numpy.ndarray: Transformed apex coordinates
        """
        rotation_matrix = np.array([
            [np.cos(heading_rad), -np.sin(heading_rad), 0],
            [np.sin(heading_rad), np.cos(heading_rad), 0],
            [0, 0, 1]
        ])
        apex_transformed = np.dot(rotation_matrix, np.array(apex).reshape(-1, 1)).flatten()
        return apex_transformed

    @staticmethod
    def __generate_distributed_values(min_val, max_val, num_points, distribution_type, is_centered_property=False):
        """
        Generate values based on the specified distribution type.
        
        Distribution types:
            - "Normal Distribution": Uniform random (flat distribution)
            - "Parabel": Beta(0.5, 0.5) - U-shaped (more at edges)
            - "Exponentiell": Exponential - more at start or center
            
        Args:
            min_val: Minimum value for the range
            max_val: Maximum value for the range
            num_points: Number of points to generate
            distribution_type: Type of distribution ("Normal Distribution", "Parabel", "Exponentiell")
            is_centered_property: If True, center the distribution (for angles)
                                If False, start from zero (for distance)
                                
        Returns:
            numpy.ndarray: Array of generated values
        """
        range_val = max_val - min_val
        
        # 1. "Normal Distribution" -> Uniform (evenly distributed)
        if distribution_type == "Normal Distribution":
            return np.random.uniform(min_val, max_val, num_points)
            
        # 2. "Parabel" -> Beta(0.5, 0.5) creates U-shape (more at edges)
        elif distribution_type == "Parabel":
            raw = np.random.beta(0.5, 0.5, num_points)
            return min_val + raw * range_val
            
        # 3. "Exponentiell" -> Exponential distribution
        elif distribution_type == "Exponentiell":
            if is_centered_property:
                # For Y-axis (angles): density should be highest in the CENTER
                # Use Normal distribution (Gaussian) which is dense in the middle
                mean = (min_val + max_val) / 2
                sigma = range_val / 6  # Controls spread width
                values = np.random.normal(mean, sigma, num_points)
                return np.clip(values, min_val, max_val)
            else:
                # For X-axis (distance): density should be highest at START (0)
                # Use exponential distribution
                scale = range_val / 4  # Controls how fast density falls off
                values = np.random.exponential(scale, num_points)
                # Shift to start at min_val (usually 0 for distance)
                values = values + min_val
                # Clip to max to prevent points beyond the range
                return np.clip(values, min_val, max_val)
        
        # Fallback: uniform distribution
        return np.random.uniform(min_val, max_val, num_points)

    @staticmethod
    def generateCone(apex, lateral_angle_left, lateral_angle_right, vertical_min_angle,
                     vertical_max_angle, max_distance, num_points, heading, aircraftOrientationAngles=None,
                     distribution_settings=None):
        """
        Generate a point cloud within a cone volume.
        
        This is the main method for generating sampling points. It creates points
        distributed in a 3D cone shape, with the apex as the tip and the cone
        spreading out based on the lateral and vertical angles.
        
        Args:
            apex: Tuple (x, y, z) - tip of the cone in local coordinates
            lateral_angle_left: Left spread angle in degrees (negative for left)
            lateral_angle_right: Right spread angle in degrees (positive for right)
            vertical_min_angle: Minimum vertical angle in degrees (downward)
            vertical_max_angle: Maximum vertical angle in degrees (upward)
            max_distance: Maximum distance from apex in meters
            num_points: Number of points to generate
            heading: Runway heading in degrees (affects coordinate transformation)
            aircraftOrientationAngles: Optional dict with pitch/yaw/roll ranges
            distribution_settings: Optional dict with distribution type and axis settings
            
        Returns:
            tuple: (points_array, apex_transformed, orientation_dict)
                - points_array: Nx3 numpy array of point coordinates
                - apex_transformed: Transformed apex coordinates
                - orientation_dict: Dict with 'pitch', 'yaw', 'roll' arrays
        """
        # Convert angles from degrees to radians
        heading_rad = np.radians(heading)
        lateral_left_rad = np.radians(lateral_angle_left)
        lateral_right_rad = np.radians(lateral_angle_right)
        vertical_min_rad = np.radians(vertical_min_angle)
        vertical_max_rad = np.radians(vertical_max_angle)

        # Transform the apex based on heading
        apex_transformed = SamplingPointGenerator.__transformAimingPoint(apex, heading_rad)

        # Rotation matrix for alignment along heading direction
        rotation_matrix = np.array([
            [np.cos(heading_rad), -np.sin(heading_rad), 0],
            [np.sin(heading_rad),  np.cos(heading_rad), 0],
            [0, 0, 1]
        ])

        # --- Distribution Settings ---
        dist_type = "Normal Distribution"
        apply_x = False
        apply_y = False

        if distribution_settings:
            dist_type = distribution_settings.get("type", "Normal Distribution")
            apply_x = distribution_settings.get("apply_x", False)
            apply_y = distribution_settings.get("apply_y", False)

        # 1. Distances (X-axis / forward direction)
        # is_centered_property=False -> For "Exponential", start at 0 (apex)
        if apply_x:
            distances = SamplingPointGenerator.__generate_distributed_values(
                0, max_distance, num_points, dist_type, is_centered_property=False
            )
        else:
            distances = np.random.uniform(0, max_distance, num_points)

        # 2. Horizontal angles (Y-axis / lateral spread)
        # is_centered_property=True -> For "Exponential", density in the center
        if apply_y:
            theta = SamplingPointGenerator.__generate_distributed_values(
                lateral_left_rad, lateral_right_rad, num_points, dist_type, is_centered_property=True
            )
        else:
            theta = np.random.uniform(lateral_left_rad, lateral_right_rad, num_points)

        # 3. Vertical angles (Z-axis / height) - currently uniform
        phi = np.random.uniform(vertical_min_rad, vertical_max_rad, num_points)

        # Convert polar coordinates to Cartesian
        x = distances
        y = distances * np.sin(theta)
        z = distances * np.sin(phi)

        # Combine into a single array
        points = np.vstack((x, y, z)).T

        # Apply rotation to align with heading
        points_rotated = points @ rotation_matrix.T

        # Translate by the transformed apex
        points_rotated += apex_transformed

        # Generate random aircraft orientations (pitch, yaw, roll)
        pitchValue = np.random.uniform(
            aircraftOrientationAngles["pitchMin"], 
            aircraftOrientationAngles["pitchMax"], 
            num_points
        )
        yawValue = np.random.uniform(
            aircraftOrientationAngles["yawMin"], 
            aircraftOrientationAngles["yawMax"], 
            num_points
        )
        rollValue = np.random.uniform(
            aircraftOrientationAngles["rollMin"], 
            aircraftOrientationAngles["rollMax"], 
            num_points
        )

        # Ensure pitchValue is an array (fix for scalar values)
        if np.isscalar(pitchValue):
            pitchValue = np.full(num_points, pitchValue)

        randomAircraftOrientation = {
            "pitch": pitchValue,
            "yaw": yawValue,
            "roll": rollValue
        }

        return points_rotated, apex_transformed, randomAircraftOrientation
