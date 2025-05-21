import numpy as np

class SamplingPointGenerator:
    def __init__(self):
        super().__init__()

    @staticmethod
    def __transformAimingPoint(apex, heading_rad):
        """
        Rotiert den Apex entsprechend dem Heading um die Z-Achse.
        :param apex: Tuple oder Liste (x, y, z) des Apex
        :param heading: Heading in Grad
        :return: Transformierter Apex als numpy-Array
        """
        rotation_matrix = np.array([
            [np.cos(heading_rad), -np.sin(heading_rad), 0],
            [np.sin(heading_rad), np.cos(heading_rad), 0],
            [0, 0, 1]
        ])
        apex_transformed = np.dot(rotation_matrix, np.array(apex).reshape(-1, 1)).flatten()
        return apex_transformed

    @staticmethod
    def generateCone(apex, lateral_angle_left, lateral_angle_right, vertical_min_angle,
                     vertical_max_angle, max_distance, num_points, heading):
        """
        Generiert eine Punktwolke, bei der der Kegel entlang der Landebahn-Ausrichtung (Heading) liegt.
        Horizontalwinkel (theta) in der Ebene der Landebahn, Vertikalwinkel (phi) senkrecht zur Landebahn.

        :param apex: Spitze des Kegels (x, y, z)
        :param lateral_angle_left: Lateraler Winkel (links) in Grad
        :param lateral_angle_right: Lateraler Winkel (rechts) in Grad
        :param vertical_min_angle: Vertikale Minimalwinkel in Grad
        :param vertical_max_angle: Vertikale Maxwinkel in Grad
        :param max_distance: Maximale Distanz der Punkte
        :param inclination: Neigung des Kegels (Pitch, Yaw, Roll) in Grad
        :param num_points: Anzahl der Punkte
        :param heading: Heading der Landebahn in Grad
        :return: Punkte des Kegels als numpy-Array (N x 3)
        """
        # Konvertiere Winkel in Radiant
        heading_rad = np.radians(heading)
        lateral_left_rad = np.radians(lateral_angle_left)
        lateral_right_rad = np.radians(lateral_angle_right)
        vertical_min_rad = np.radians(vertical_min_angle)
        vertical_max_rad = np.radians(vertical_max_angle)

        # Transformiere den Apex
        apex_transformed = SamplingPointGenerator.__transformAimingPoint(apex, heading_rad)

        # Rotationsmatrix für die Ausrichtung entlang des Headings
        rotation_matrix = np.array([
            [np.cos(heading_rad), -np.sin(heading_rad), 0],
            [np.sin(heading_rad), np.cos(heading_rad), 0],
            [0, 0, 1]
        ])

        # Generiere die Distanzen und zufällige Winkel
        distances = np.linspace(0, max_distance, num_points)
        theta = np.random.uniform(lateral_left_rad, lateral_right_rad, num_points)  # Horizontale Winkel (Y-Z-Ebene)
        phi = np.random.uniform(vertical_min_rad, vertical_max_rad, num_points)  # Vertikalwinkel zur X-Achse

        # Punkte in kartesischen Koordinaten berechnen
        x = distances  # Kegellängsachse entlang der X-Achse
        y = distances * np.sin(theta)  # Horizontale Verteilung (Y-Achse)
        z = distances * np.sin(phi)  # Vertikale Verteilung (Z-Achse)

        # Punkte zu einem Array zusammenführen
        points = np.vstack((x, y, z)).T

        # Punkte rotieren
        points_rotated = points @ rotation_matrix.T

        # Punkte um den rotierten Apex verschieben
        points_rotated += apex_transformed

        return points_rotated, apex_transformed
