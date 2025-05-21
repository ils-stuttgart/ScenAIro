import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import json


class RunwayTaggingEngine:
    def __init__(self):
        super().__init__()

    

    # ---------------------
    # camera- and pixel calculations
    # ---------------------

    def rotate3DPoint(self, vector, pitch, yaw, roll):
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        roll_rad = np.radians(roll)

        R_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)]
        ])

        R_yaw = np.array([
            [-np.cos(yaw_rad), np.sin(yaw_rad), 0],
            [-np.sin(yaw_rad), -np.cos(yaw_rad), 0],
            [0, 0, 1]
        ])

        R_roll = np.array([
            [np.cos(roll_rad), 0, np.sin(roll_rad)],
            [0, 1, 0],
            [-np.sin(roll_rad), 0, np.cos(roll_rad)]
        ])

        R = R_yaw @ R_pitch @ R_roll
        return R @ vector

    def calculatePixelCoordinates(self, point, horizontal_fov, vertical_fov, screen_width, screen_height):
        """
        Berechnet die Pixelkoordinaten eines Punktes im Kamerabild.
        
        point: (x, y, z) mit:
        x = Distanz zum Punkt in Kamerarichtung
        y = Abweichung nach links/rechts
        z = Abweichung nach oben/unten
        horizontal_fov, vertical_fov: Sichtfeld der Kamera in Grad
        screen_width, screen_height: Größe des Bildschirms in Pixeln
        
        Rückgabe:
        (x_pixel, y_pixel): Bildschirmkoordinaten
        (x_distance, y_distance): Abweichung vom Bildmittelpunkt
        """

        horizontal_fov_radians = np.radians(horizontal_fov)
        vertical_fov_radians = np.radians(vertical_fov)


        # Berechnung der Fokuslängen basierend auf dem FOV
        f_horizontal = (screen_width / 2) / np.tan(horizontal_fov_radians / 2)
        f_vertical = ((screen_height / 2) / np.tan(vertical_fov_radians / 2))


        #self.sm = SimConnect()
        #self.aq = AircraftRequests(self.sm, _time=2000)
        #altitudeFromSim = (self.aq.get("PLANE_ALTITUDE")) * 0.3048

        # Extrahierte Koordinaten: x = Blickrichtung, y = links/rechts, z = oben/unten
        x, y, z = point
        distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Projektion auf die Bildkoordinaten
        u = y * (f_horizontal / x)
        v = z * (f_vertical / x)

        # Berechnung der Pixelkoordinaten auf dem Bildschirm
        x_pixel = ((screen_width / 2) + u) 
        y_pixel =((screen_height / 2) - v) 

        print(f"Pixelkoordinaten: ({x_pixel}, {y_pixel})")

        return (int(round(x_pixel)), int(round(y_pixel)))



    # ---------------------
    # Visualisation for Testing
    # ---------------------

    def saveAnnotation(self, screenshot_name, structured_objects, image_width, image_height, horizontal_fov_degrees, vertical_fov_degrees, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        annotations = []
        
        for obj_id, obj in enumerate(structured_objects):
            pixel_coords = []
            for label, point in zip(["A", "B", "C", "D"], [obj.A, obj.B, obj.C, obj.D]):
                rotated_point = self.rotate3DPoint(point, 0, 0, 0)
                x_pixel, y_pixel = self.calculatePixelCoordinates(
                    rotated_point, horizontal_fov_degrees, vertical_fov_degrees, image_width, image_height
                )
                pixel_coords.extend([x_pixel, y_pixel])
            
            bbox_x = min(pixel_coords[0::2])
            bbox_y = min(pixel_coords[1::2])
            bbox_width = max(pixel_coords[0::2]) - bbox_x
            bbox_height = max(pixel_coords[1::2]) - bbox_y
            
            annotation = {
                "id": obj_id,
                "image_id": screenshot_name,
                "category_id": 1,
                "bbox": [bbox_x, bbox_y, bbox_width, bbox_height],
                "segmentation": [pixel_coords],
                "area": bbox_width * bbox_height,
                "iscrowd": 0
            }
            annotations.append(annotation)
        
        coco_format = {
            "images": [
                {
                    "file_name": f"{screenshot_name[:-4]}.png",
                    "id": screenshot_name,
                    "width": image_width,
                    "height": image_height
                }
            ],
            "annotations": annotations,
            "categories": [
                {"id": 1, "name": "runway", "supercategory": "infrastructure"}
            ]
        }
        
        json_filename = os.path.join(output_dir, f"{screenshot_name[:-4]}.json")
        with open(json_filename, "w") as json_file:
            json.dump(coco_format, json_file, indent=4)
        
        print(f"COCO-Annotation gespeichert: {json_filename}")

    def doOverlayLabelsOnImage(self, image_path, output_path, structured_objects,
                               horizontal_fov_degrees, vertical_fov_degrees,
                               screen_width, screen_height,
                               cam_pitch, cam_yaw, cam_roll):

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Bild konnte nicht geladen werden: {image_path}")

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        if not structured_objects:
            raise ValueError("Keine strukturierten Objekte vorhanden!")
        
        # safe Annotation as JSON
        self.saveAnnotation(os.path.basename(image_path), structured_objects, screen_width, screen_height, horizontal_fov_degrees, vertical_fov_degrees, os.path.dirname(output_path))

        # Alle Runway-Eckpunkte für jedes StructuredObject zeichnen
    
        for obj in structured_objects:
            pixel_coords = []

            for label, point in zip(["A", "B", "C", "D"], [obj.A, obj.B, obj.C, obj.D]):
                # point, roll, yaw, pitch
                rotated_point = self.rotate3DPoint(point, 0, 0, 0)

                (x_pixel, y_pixel) = self.calculatePixelCoordinates(
                    rotated_point, horizontal_fov_degrees, vertical_fov_degrees, screen_width, screen_height
                )
                x_pixel = int(round(x_pixel))
                y_pixel = int(round(y_pixel))

                if 0 <= x_pixel < image.shape[1] and 0 <= y_pixel < image.shape[0]:
                    cv2.circle(image, (x_pixel, y_pixel), radius=3, color=(0, 0, 255), thickness=-1)
                    cv2.putText(image, label, (x_pixel + 5, y_pixel - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    pixel_coords.append((x_pixel, y_pixel))

            # Zeichne ein Rechteck, falls mindestens 4 Punkte vorhanden sind:
            if len(pixel_coords) == 4:
                cv2.line(image, pixel_coords[0], pixel_coords[1], (0, 255, 0), 1)  # A -> B
                cv2.line(image, pixel_coords[1], pixel_coords[2], (0, 255, 0), 1)  # B -> C
                cv2.line(image, pixel_coords[2], pixel_coords[3], (0, 255, 0), 1)  # C -> D
                cv2.line(image, pixel_coords[3], pixel_coords[0], (0, 255, 0), 1)  # D -> A

        if not cv2.imwrite(output_path, image):
            raise IOError(f"Fehler beim Speichern des Bildes unter: {output_path}")

        print(f"Speichere Bild unter: {output_path}")
    
