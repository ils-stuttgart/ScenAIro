"""
RunwayCornerAnnotationStruct - Runway Corner Annotation Data Structures
========================================================================

This module provides data structures for runway corner annotations used
in the ScenAIro labeling system.

The module defines:
    - Point: A 3D coordinate class
    - Angle: Aircraft orientation (pitch, bank, roll)
    - StructuredObject: A collection of runway corners with orientation
    - RunwayCornerAnnotationStruct: Calculator for corner positions relative to aircraft

Author: ScenAIro Team
"""

import numpy as np
import math


class Point:
    """
    Represents a 3D point in space.
    
    Attributes:
        x: X coordinate (default 0.0)
        y: Y coordinate (default 0.0)
        z: Z coordinate (default 0.0)
    """
    
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"
    
    def __sub__(self, other):
        """Subtract another Point to get a new Point representing the vector between them."""
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)


class Angle:
    """
    Represents aircraft orientation angles.
    
    Attributes:
        pitch: Rotation around Y-axis (nose up/down) in degrees
        bank: Rotation around X-axis (wing tilt/roll) in degrees
        roll: Alias for bank (rotation around longitudinal axis)
    """
    
    def __init__(self, pitch: float, bank: float, roll: float):
        self.pitch = pitch
        self.bank = bank
        self.roll = roll

    def __repr__(self):
        return f"Angle(pitch={self.pitch}, bank={self.bank}, roll={self.roll})"


class StructuredObject:
    """
    Represents a structured annotation object containing runway corner positions.
    
    The corners are labeled A, B, C, D representing:
        A: Top-left corner (from approach perspective)
        B: Top-right corner
        C: Bottom-right corner
        D: Bottom-left corner
    
    Attributes:
        A, B, C, D: Corner positions as tuples (x, y, z)
        Angles: Aircraft orientation (pitch, yaw, roll)
    """
    
    def __init__(self, A, B, C, D, Angles):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Angles = Angles

    def __repr__(self):
        return (
            f"StructuredObject(A={self.A}, B={self.B}, C={self.C}, D={self.D}, "
            f"Angles={self.Angles})"
        )


class RunwayCornerAnnotationStruct:
    """
    Calculates runway corner positions relative to an aircraft position.
    
    This class transforms runway corner coordinates from the world frame to
    the aircraft's local reference frame, accounting for runway heading.
    This is essential for accurate labeling in the generated training data.
    """

    def __init__(self):
        """Initialize with default zero values."""
        zero_point = (0.0, 0.0, 0.0)
        dummy_angle = Angle(0.0, 0.0, 0.0)  # Angles set to zero
        self.structuredObjects = StructuredObject(zero_point, zero_point, zero_point, zero_point, dummy_angle)
        self.angles = dummy_angle
        self.point = Point(0.0, 0.0, 0.0)

    def calculateAirplane2RunwayCornerVector(self, airplaneCoord, cornerCoord, runwayHeading, centerHeight):
        """
        Calculate the 2D displacement between aircraft position and runway corner with rotation.
        
        This method computes the vector from the aircraft to a specific runway corner,
        accounting for the runway heading by applying a rotation transformation.
        
        Args:
            airplaneCoord: Aircraft position as (x, y, z) tuple
            cornerCoord: Runway corner position as (x, y, z) tuple
            runwayHeading: Runway heading in degrees
            centerHeight: Reference center height for altitude calculation
            
        Returns:
            tuple: (x, y, z) displaced coordinates from aircraft to corner
        """
        # Calculate height difference (z-axis) - remains unchanged by rotation
        zDistance = cornerCoord[2] - airplaneCoord[2] - centerHeight

        # Calculate x and y distances before rotation
        xDistance = cornerCoord[0] - airplaneCoord[0]
        yDistance = cornerCoord[1] - airplaneCoord[1]

        # Original point coordinates as a vector
        original_coords = np.array([xDistance, yDistance, zDistance])

        # Create rotation matrix based on runway heading
        alpha = np.radians(-runwayHeading)
        R_alpha = np.array([
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha),  math.cos(alpha), 0],
            [0, 0, 1]
        ])

        # Apply rotation
        rotated_coords = R_alpha @ original_coords

        return (rotated_coords[0], rotated_coords[1], rotated_coords[2])

    def calculateAirplane2RunwayCornerStructure(self, point, runway_corners, angles, runwayHeading, centerHeight):
        """
        Calculate the complete structure of all four runway corners relative to the aircraft.
        
        Args:
            point: Aircraft position as (x, y, z)
            runway_corners: Dict with keys 'top_left', 'top_right', 'bottom_left', 'bottom_right'
            angles: Tuple of (pitch, yaw, roll) angles
            runwayHeading: Runway heading in degrees
            centerHeight: Reference center height
            
        Returns:
            list: List containing one StructuredObject with all four corner positions
        """
        objects = []
        
        # Calculate relative position for each corner
        A = self.calculateAirplane2RunwayCornerVector(point, runway_corners["top_left"], runwayHeading, centerHeight)
        B = self.calculateAirplane2RunwayCornerVector(point, runway_corners["top_right"], runwayHeading, centerHeight)
        C = self.calculateAirplane2RunwayCornerVector(point, runway_corners["bottom_right"], runwayHeading, centerHeight)
        D = self.calculateAirplane2RunwayCornerVector(point, runway_corners["bottom_left"], runwayHeading, centerHeight)

        structured_object = StructuredObject(A, B, C, D, angles)
        objects.append(structured_object)

        return objects
