import numpy as np

class SamplingPointGenerator:
    def __init__(self):
        super().__init__()

    @staticmethod
    def __transformAimingPoint(apex, heading_rad):
        """
        Rotiert den Apex entsprechend dem Heading um die Z-Achse.
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
        Generiert Werte basierend auf der gewünschten Verteilung.
        
        :param is_centered_property: 
            True = Wir verteilen Winkel (wir wollen die Dichte in der Mitte/Achse haben).
            False = Wir verteilen Distanz (wir wollen die Dichte am Start/Apex haben).
        """
        range_val = max_val - min_val
        
        # 1. "Normal Distribution" -> Soll jetzt UNIFORM (gleichmäßig) sein
        if distribution_type == "Normal Distribution":
            return np.random.uniform(min_val, max_val, num_points)
            
        # 2. "Parabel" -> Soll an den RÄNDERN mehr sein (U-Form)
        elif distribution_type == "Parabel":
            # Beta(0.5, 0.5) erzeugt eine "Badewannenkurve" (viel außen, wenig innen)
            raw = np.random.beta(0.5, 0.5, num_points)
            return min_val + raw * range_val
            
        # 3. "Exponentiell" -> Soll am Anfang bzw. in der Mitte viel sein
        elif distribution_type == "Exponentiell":
            if is_centered_property:
                # Fall Y-Achse (Winkel): Dichte soll in der MITTE hoch sein
                # Wir nutzen hier eine Normalverteilung (Gauß), da diese in der Mitte dicht ist
                mean = (min_val + max_val) / 2
                sigma = range_val / 6  # Breite der Streuung
                values = np.random.normal(mean, sigma, num_points)
                return np.clip(values, min_val, max_val)
            else:
                # Fall X-Achse (Distanz): Dichte soll am START (0) hoch sein
                # Wir nutzen eine exponentielle Verteilung
                scale = range_val / 4  # Steuert wie schnell die Dichte abfällt
                values = np.random.exponential(scale, num_points)
                # Verschieben auf min_val (meist 0 bei Distanz)
                values = values + min_val
                # Abschneiden bei max, damit keine Punkte zu weit weg liegen
                return np.clip(values, min_val, max_val)
            
        # Fallback
        return np.random.uniform(min_val, max_val, num_points)

    @staticmethod
    def generateCone(apex, lateral_angle_left, lateral_angle_right, vertical_min_angle,
                     vertical_max_angle, max_distance, num_points, heading, aircraftOrientationAngles=None,
                     distribution_settings=None):
        """
        Generiert eine Punktwolke mit spezifischen Verteilungsregeln.
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

        # --- VERTEILUNGSAUSWERTUNG ---
        dist_type = "Normal Distribution"
        apply_x = False
        apply_y = False

        if distribution_settings:
            dist_type = distribution_settings.get("type", "Normal Distribution")
            apply_x = distribution_settings.get("apply_x", False)
            apply_y = distribution_settings.get("apply_y", False)

        # 1. Distanzen (X-Achse)
        # is_centered_property=False -> Bei "Exponentiell" Start bei 0 (Apex)
        if apply_x:
            distances = SamplingPointGenerator.__generate_distributed_values(
                0, max_distance, num_points, dist_type, is_centered_property=False
            )
        else:
            distances = np.random.uniform(0, max_distance, num_points)

        # 2. Horizontale Winkel (Y-Achse / Breite)
        # is_centered_property=True -> Bei "Exponentiell" Dichte in der Mitte
        if apply_y:
            theta = SamplingPointGenerator.__generate_distributed_values(
                lateral_left_rad, lateral_right_rad, num_points, dist_type, is_centered_property=True
            )
        else:
            theta = np.random.uniform(lateral_left_rad, lateral_right_rad, num_points)

        # 3. Vertikale Winkel (Z-Achse / Höhe) - bleibt vorerst uniform
        phi = np.random.uniform(vertical_min_rad, vertical_max_rad, num_points)

        # Punkte in kartesischen Koordinaten berechnen
        x = distances
        y = distances * np.sin(theta)
        z = distances * np.sin(phi)

        # Punkte zu einem Array zusammenführen
        points = np.vstack((x, y, z)).T

        # Punkte rotieren
        points_rotated = points @ rotation_matrix.T

        # Punkte um den rotierten Apex verschieben
        points_rotated += apex_transformed

        # Generiere die zufälligen Orientierungen
        pitchValue = np.random.uniform(aircraftOrientationAngles["pitchMin"], aircraftOrientationAngles["pitchMax"], num_points)
        yawValue = np.random.uniform(aircraftOrientationAngles["yawMin"], aircraftOrientationAngles["yawMax"], num_points)
        rollValue = np.random.uniform(aircraftOrientationAngles["rollMin"], aircraftOrientationAngles["rollMax"], num_points)

        # Sicherstellen, dass pitchValue ein Array ist (Fix für "not subscriptable")
        if np.isscalar(pitchValue):
            pitchValue = np.full(num_points, pitchValue)

        randomAircraftOrientation = {
            "pitch": pitchValue,
            "yaw": yawValue,
            "roll": rollValue
        }

        return points_rotated, apex_transformed, randomAircraftOrientation