"""
Runway Tagging Engine Module

This module provides functionality for projecting 3D runway corner coordinates
onto 2D image coordinates and generating COCO-format annotations for training
machine learning models.
"""

from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import json


class RunwayTaggingEngine:
    """
    Engine for tagging runway images with corner annotations.
    
    Handles 3D-to-2D projection of runway corner points and generates
    COCO-format annotation files for machine learning datasets.
    """
    
    def __init__(self):
        super().__init__()

    @staticmethod
    def _make_json_safe(obj):
        """
        Convert NumPy arrays and scalars to JSON-compatible Python types.
        
        Recursively processes nested dictionaries and lists to ensure all
        NumPy types are converted to native Python types.
        
        Args:
            obj: Object to convert (can be numpy array, scalar, dict, or list)
            
        Returns:
            JSON-serializable version of the input object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            # Handle NumPy scalars (np.float32, np.int64, etc.)
            return obj.item()
        if isinstance(obj, dict):
            return {k: RunwayTaggingEngine._make_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [RunwayTaggingEngine._make_json_safe(v) for v in obj]
        return obj

    # ---------------------
    # Camera and Pixel Calculations
    # ---------------------

    def rotate3DPoint(self, vector, pitch, yaw, roll):
        """
        Rotate a 3D point using aircraft orientation angles.
        
        Applies rotation to transform world coordinates to camera coordinates.
        The sign convention may differ between the simulator and the rotation matrices.
        
        Args:
            vector: 3D point as numpy array [x, y, z]
            pitch: Rotation around Y-axis (nose up/down) in degrees
            yaw: Rotation around Z-axis (turning left/right) in degrees
            roll: Rotation around X-axis (banking) in degrees
            
        Returns:
            Rotated 3D point as numpy array (in camera coordinates)
            
        Note:
            Coordinate system convention:
            - X-axis: Depth (forward direction)
            - Y-axis: Left/right
            - Z-axis: Up/down (height)
        """
        # If all angles are zero, return the vector unchanged
        if pitch == 0 and yaw == 0 and roll == 0:
            return vector
        
        # Convert angles from degrees to radians
        # To transform world coordinates to camera coordinates, we need the INVERSE
        # of the camera rotation. For rotation matrices, inverse = transpose.
        # Yaw and roll have opposite sign convention in the simulator, but pitch
        # needs to remain as-is for correct visual alignment.
        p = np.radians(pitch)      # Keep pitch as-is
        y = np.radians(-yaw)       # Negate yaw for inverse
        r = np.radians(roll)       # Keep roll as-is 
        
        # Rotation matrix around X-axis (Roll)
        mat_roll = np.array([
            [1, 0, 0],
            [0, np.cos(r), -np.sin(r)],
            [0, np.sin(r), np.cos(r)]
        ])

        # Rotation matrix around Y-axis (Pitch)
        mat_pitch = np.array([
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)]
        ])

        # Rotation matrix around Z-axis (Yaw)
        mat_yaw = np.array([
            [np.cos(y), -np.sin(y), 0],
            [np.sin(y), np.cos(y), 0],
            [0, 0, 1]
        ])

        # Combined rotation - REVERSED order for inverse: roll @ pitch @ yaw
        R = mat_roll @ mat_pitch @ mat_yaw
        
        result = R @ vector
        print(f"[DEBUG] rotate3DPoint: input={vector}, angles=({pitch},{yaw},{roll}), output={result}")
        
        return result

    def calculatePixelCoordinates(self, point, horizontal_fov, vertical_fov, 
                                   screen_width, screen_height):
        """
        Calculate pixel coordinates from a 3D point in camera space.
        
        Projects a 3D point onto a 2D image plane using perspective projection
        based on the camera's field of view and screen dimensions.
        
        Args:
            point: Tuple (x, y, z) where:
                - x: Distance to point in camera direction (depth, positive = in front)
                - y: Deviation left/right
                - z: Deviation up/down
            horizontal_fov: Horizontal field of view in degrees
            vertical_fov: Vertical field of view in degrees
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            
        Returns:
            Tuple (x_pixel, y_pixel): Screen coordinates in pixels
        """
        # Convert FOV from degrees to radians
        horizontal_fov_radians = np.radians(horizontal_fov)
        vertical_fov_radians = np.radians(vertical_fov)

        # Calculate focal lengths based on FOV
        f_horizontal = (screen_width / 2) / np.tan(horizontal_fov_radians / 2)
        f_vertical = (screen_height / 2) / np.tan(vertical_fov_radians / 2)

        # Extract coordinates: x = view direction, y = left/right, z = up/down
        x, y, z = point

        # Project onto image coordinates using perspective projection
        u = y * (f_horizontal / x)
        v = z * (f_vertical / x)

        # Convert to pixel coordinates (origin at top-left of screen)
        x_pixel = (screen_width / 2) + u
        y_pixel = (screen_height / 2) - v

        return (int(round(x_pixel)), int(round(y_pixel)))

    # ---------------------
    # Visualization for Testing
    # ---------------------

    def saveAnnotation(self, screenshot_name, structured_objects, image_width, 
                       image_height, horizontal_fov_degrees, vertical_fov_degrees, 
                       output_dir, airport_data=None, cone_data=None, geo_point=None, 
                       generated_point=None, aircraft_orientation=None, daytime=None,
                       weather_data=None, distribution_data=None):
        """
        Save runway annotations in COCO format.
        
        Generates a JSON file containing bounding boxes and segmentation
        polygons for runway detection training datasets.
        
        Args:
            screenshot_name: Name of the screenshot file
            structured_objects: List of objects with corner points (A, B, C, D)
            image_width: Image width in pixels
            image_height: Image height in pixels
            horizontal_fov_degrees: Horizontal field of view in degrees
            vertical_fov_degrees: Vertical field of view in degrees
            output_dir: Directory to save the annotation file
            airport_data: Optional airport metadata
            cone_data: Optional landing approach cone data
            geo_point: Optional aircraft geographic position
            generated_point: Optional 3D position relative to runway
            aircraft_orientation: Optional (pitch, yaw, roll) tuple
            daytime: Optional daytime information
            weather_data: Optional weather information
            distribution_data: Optional distribution metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        annotations = []

        # Process each structured object (runway)
        for obj_id, obj in enumerate(structured_objects):
            pixel_coords = []
            
            # Project each corner point to pixel coordinates
            for label, point in zip(["A", "B", "C", "D"], [obj.A, obj.B, obj.C, obj.D]):
                rotated_point = self.rotate3DPoint(point, *aircraft_orientation)
                x_pixel, y_pixel = self.calculatePixelCoordinates(
                    rotated_point, horizontal_fov_degrees, vertical_fov_degrees, 
                    image_width, image_height
                )
                pixel_coords.extend([x_pixel, y_pixel])
            
            # Calculate bounding box from corner points
            bbox_x = min(pixel_coords[0::2])
            bbox_y = min(pixel_coords[1::2])
            bbox_width = max(pixel_coords[0::2]) - bbox_x
            bbox_height = max(pixel_coords[1::2]) - bbox_y
            
            # Create COCO annotation entry
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
        
        # Build complete COCO format structure
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

        # Add optional metadata
        if airport_data is not None:
            coco_format["runway_data"] = airport_data

        if cone_data is not None:
            coco_format["landing_approach_cone"] = cone_data

        if geo_point is not None:
            coco_format["position_of_aircraft"] = geo_point

        if generated_point is not None:
            # Extract coordinates from numpy array or tuple/list
            if isinstance(generated_point, np.ndarray):
                gx, gy, gz = generated_point.tolist()
            else:
                gx, gy, gz = generated_point

            # Calculate ground distance using Pythagorean theorem (x, y plane)
            distance_ground = float(np.sqrt(gx**2 + gy**2))

            # Height difference (z component)
            altitude_difference = float(gz)

            distance_ground = round(distance_ground, 2)
            altitude_difference = round(altitude_difference, 2)

            coco_format["distance_aircraft_2_runway"] = {
                "ground_distance_in_meters": distance_ground,
                "altitude_difference_in_meters": altitude_difference,
            }

        if aircraft_orientation is not None:
            coco_format["aircraft_orientation"] = {
                "pitch": aircraft_orientation[0],
                "yaw": aircraft_orientation[1],
                "roll": aircraft_orientation[2]
            }

        if daytime is not None:
            coco_format["daytime"] = daytime

        if weather_data is not None:
            coco_format["weather"] = weather_data

        # Save to JSON file
        json_filename = os.path.join(output_dir, f"{screenshot_name[:-4]}.json")

        # Convert all NumPy types to JSON-compatible Python types
        coco_format_safe = self._make_json_safe(coco_format)

        with open(json_filename, "w") as json_file:
            json.dump(coco_format_safe, json_file, indent=4)

        print(f"COCO annotation saved: {json_filename}")

    def doOverlayLabelsOnImage(self, image_path, output_path, structured_objects,
                                horizontal_fov_degrees, vertical_fov_degrees,
                                screen_width, screen_height, cam_pitch, cam_yaw, 
                                cam_roll, airport_data=None, cone_data=None,
                                geo_point=None, generated_point=None, daytime=None,
                                weather_data=None, distribution_data=None,
                                excludeImg=False):
        """
        Overlay runway corner labels on an image and save annotations.
        
        Projects 3D runway corner points onto the 2D image, draws visual
        markers, and saves COCO-format annotations to a JSON file.
        
        Args:
            image_path: Path to the input screenshot
            output_path: Path for the output annotated image
            structured_objects: List of runway objects with corner points
            horizontal_fov_degrees: Horizontal field of view in degrees
            vertical_fov_degrees: Vertical field of view in degrees
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            cam_pitch: Camera pitch angle in degrees
            cam_yaw: Camera yaw angle in degrees
            cam_roll: Camera roll angle in degrees
            airport_data: Optional airport metadata
            cone_data: Optional landing approach cone data
            geo_point: Optional aircraft geographic position
            generated_point: Optional 3D position relative to runway
            daytime: Optional daytime information
            weather_data: Optional weather information
            distribution_data: Optional distribution metadata
            excludeImg: If True, only save JSON without creating overlay image
            
        Raises:
            ValueError: If no structured objects are provided
            FileNotFoundError: If the input image cannot be loaded
            IOError: If the output image cannot be saved
        """
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        if not structured_objects:
            raise ValueError("No structured objects provided!")

        # Always save JSON annotations, regardless of image generation
        self.saveAnnotation(
            os.path.basename(image_path),
            structured_objects,
            screen_width,
            screen_height,
            horizontal_fov_degrees,
            vertical_fov_degrees,
            os.path.dirname(output_path),
            airport_data=airport_data,
            cone_data=cone_data,
            geo_point=geo_point,
            generated_point=generated_point,
            aircraft_orientation=(cam_pitch, cam_yaw, cam_roll),
            daytime=daytime,
            weather_data=weather_data,
            distribution_data=distribution_data
        )

        # Skip image overlay if excludeImg is True
        if excludeImg:
            print("[INFO] excludeImg=True - No overlay image created, only JSON saved.")
            return

        # Load the input image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Draw all runway corner points for each structured object
        for obj in structured_objects:
            pixel_coords = []

            # Project each corner point (A, B, C, D) to pixel coordinates
            for label, point in zip(["A", "B", "C", "D"], [obj.A, obj.B, obj.C, obj.D]):
                # Apply aircraft orientation to the 3D point
                rotated_point = self.rotate3DPoint(point, cam_pitch, cam_yaw, cam_roll)

                (x_pixel, y_pixel) = self.calculatePixelCoordinates(
                    rotated_point, horizontal_fov_degrees, vertical_fov_degrees, 
                    screen_width, screen_height
                )
                x_pixel = int(round(x_pixel))
                y_pixel = screen_height - int(round(y_pixel))

                # Draw marker and label if point is within image bounds
                if 0 <= x_pixel < image.shape[1] and 0 <= y_pixel < image.shape[0]:
                    cv2.circle(image, (x_pixel, y_pixel), radius=3, 
                              color=(0, 0, 255), thickness=-1)
                    cv2.putText(image, label, (x_pixel + 5, y_pixel - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    pixel_coords.append((x_pixel, y_pixel))

            # Draw rectangle connecting all 4 corner points
            if len(pixel_coords) == 4:
                cv2.line(image, pixel_coords[0], pixel_coords[1], (0, 255, 0), 1)  # A -> B
                cv2.line(image, pixel_coords[1], pixel_coords[2], (0, 255, 0), 1)  # B -> C
                cv2.line(image, pixel_coords[2], pixel_coords[3], (0, 255, 0), 1)  # C -> D
                cv2.line(image, pixel_coords[3], pixel_coords[0], (0, 255, 0), 1)  # D -> A

        # Save the annotated image
        if not cv2.imwrite(output_path, image):
            raise IOError(f"Failed to save image: {output_path}")

        print(f"Image saved: {output_path}")
