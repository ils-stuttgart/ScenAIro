import numpy as np
import math
#import ressources.constants as const

class Point:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

class Angle:
    def __init__(self, pitch: float, bank: float, roll: float):
        self.pitch = pitch
        self.bank = bank
        self.roll = roll

    def __repr__(self):
        return f"Angle(pitch={self.pitch}, bank={self.bank}, roll={self.roll})"

class StructuredObject:
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
    def __init__(self):
        zero_point = (0.0, 0.0, 0.0)
        dummy_angle = Angle(0.0, 0.0, 0.0)
        self.structuredObjects = StructuredObject(zero_point, zero_point, zero_point, zero_point, dummy_angle)
        self.angles = dummy_angle
        self.point = Point(0.0, 0.0, 0.0)

    def calculateAirplane2RunwayCornerVector(self, airplaneCoord, cornerCoord, runwayHeading, centerHeight):
        """Berechnet die 2D-Verschiebung zwischen Flugzeugposition und Runway-Ecke mit Rotation."""

        # Höhe berechnen (bleibt unverändert)
        zDistance = cornerCoord[2] - airplaneCoord[2] - centerHeight 

        # Berechnung der x und y Distanzen vor der Rotation
        xDistance = cornerCoord[0] - airplaneCoord[0]
        yDistance = cornerCoord[1] - airplaneCoord[1]

        # Ursprüngliche Punktkoordinaten als Vektor
        original_coords = np.array([xDistance, yDistance, zDistance])

        # Rotationsmatrix basierend auf vorheriger Berechnung

        alpha = np.radians(-runwayHeading)
        R_alpha = np.array([
            [math.cos(alpha), -math.sin(alpha), 0],
            [math.sin(alpha),  math.cos(alpha), 0],
            [0, 0, 1]
        ])

        # Gesamtrotation: zuerst R_beta, dann R_alpha
        R_total = R_alpha

        # Rotation anwenden
        rotated_coords = R_total @ original_coords
        #rotated_coords = original_coords

        rotated_coords[0] = rotated_coords[0] 
        rotated_coords[1] = rotated_coords[1]  

        # Ergebnisse ausgeben
        print(f"Rotated X: {rotated_coords[0]}, Rotated Y: {rotated_coords[1]}, Z: {zDistance}")

        return (rotated_coords[0], rotated_coords[1], rotated_coords[2])

    def calculateAirplane2RunwayCornerStructure(self, point, runway_corners, angles, runwayHeading, centerHeight):
        objects = []
        
        A = self.calculateAirplane2RunwayCornerVector(point, runway_corners["top_left"], runwayHeading, centerHeight)
        B = self.calculateAirplane2RunwayCornerVector(point, runway_corners["top_right"], runwayHeading, centerHeight)
        C = self.calculateAirplane2RunwayCornerVector(point, runway_corners["bottom_right"], runwayHeading, centerHeight)
        D = self.calculateAirplane2RunwayCornerVector(point, runway_corners["bottom_left"], runwayHeading, centerHeight)

        structured_object = StructuredObject(A, B, C, D, angles)
        objects.append(structured_object)

        return objects
