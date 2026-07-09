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

    # --- MSFS render-camera geometry ------------------------------------------
    # The capture camera is a RIGID OFFSET from the aircraft reference point, taken
    # from the MSFS camera-debug readout (position offset + a fixed mount pitch).
    # Modelling this physically (instead of a constant pixel shift) is what makes the
    # projection correct across near AND far frames. Defaults below are overridden by
    # config/settings.json "camera" if present.
    MSFS_PITCH_SIGN = -1.0                       # MSFS PLANE_PITCH_DEGREES positive nose-DOWN
    CAMERA_OFFSET_BODY = (-0.396, 1.021, -4.990)  # camera vs aircraft ref: [right, up, forward] m
    CAMERA_MOUNT_PITCH_DEG = -2.25               # camera boresight pitch vs airframe (deg)
    NEAR_PLANE_M = 1.0                           # near clip plane (m in front of camera)

    def __init__(self):
        super().__init__()
        # Load the camera geometry from settings (falls back to the constants above).
        self.camera_offset_body = self.CAMERA_OFFSET_BODY
        self.camera_mount_pitch_deg = self.CAMERA_MOUNT_PITCH_DEG
        try:
            from tools.SettingsManager import SettingsManager
            settings = SettingsManager()
            self.camera_offset_body = tuple(settings.get("camera", "camera_offset_body_m"))
            self.camera_mount_pitch_deg = float(settings.get("camera", "camera_mount_pitch_deg"))
        except Exception as exc:
            print(f"[RunwayTaggingEngine] Using default camera geometry ({exc})")

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

        # Transform the point from the zero-orientation "eye" frame into the frame of
        # the camera rotated by (pitch, yaw, roll). That is the INVERSE of the camera's
        # own rotation, so we must (a) negate every axis whose simulator sign convention
        # is opposite this eye frame and (b) compose the single-axis rotations in the
        # reverse order of how the aircraft orientation is built (yaw -> pitch -> roll),
        # i.e. roll @ pitch @ yaw.
        #
        # Signs verified against an independent ENU/pinhole camera model in
        # tests/test_coordinate_transforms.py:
        #   pitch: as-is  (+pitch = nose up  -> runway moves down in image)
        #   yaw:   negated (+yaw   = nose right -> runway moves left in image)
        #   roll:  negated (+roll  = bank right -> runway rotates correctly)
        p = np.radians(pitch)      # pitch: keep sign
        y = np.radians(-yaw)       # yaw: negate
        r = np.radians(-roll)      # roll: negate

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

        # Combined inverse rotation: reverse order (roll @ pitch @ yaw).
        R = mat_roll @ mat_pitch @ mat_yaw

        result = R @ vector

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

    def _cameraSpaceENU(self, north, east, up, camera_heading_deg, pitch_deg, roll_deg):
        """
        Transform an ENU vector (camera -> point) into camera space (right, up, forward).

        This is the geometry of the pinhole camera *before* the perspective divide: it
        applies the render camera's rigid offset and mount pitch (see class constants) and
        returns the point in camera axes. ``projectPointENU`` divides this by the depth to
        get pixels; the clipping helpers use it directly so they can split polygon edges on
        the near plane before any division by depth.

        Args:
            north, east, up: Vector from the camera (aircraft) to the target point, in
                metres, in a local ENU-style frame (north, east, up).
            camera_heading_deg: True heading the camera looks along (``runway_heading - 180
                + yaw``), in degrees.
            pitch_deg: Aircraft pitch value from the metadata (MSFS convention), in degrees.
            roll_deg: Aircraft roll/bank value from the metadata, in degrees.

        Returns:
            numpy array [x_c, y_c, z_c]: right, up, and forward (depth along the boresight;
            > 0 in front of the camera) coordinates in metres.
        """
        hd = np.radians(camera_heading_deg)
        th = np.radians(self.MSFS_PITCH_SIGN * pitch_deg)   # MSFS pitch: positive nose-down
        rl = np.radians(roll_deg)

        # Aircraft body basis in ENU (east, north, up).
        forward = np.array([np.sin(hd) * np.cos(th), np.cos(hd) * np.cos(th), np.sin(th)])
        right0 = np.array([np.cos(hd), -np.sin(hd), 0.0])   # right of heading, horizontal
        up0 = np.cross(right0, forward)                     # right x forward -> up
        right = np.cos(rl) * right0 + np.sin(rl) * up0
        up_vec = -np.sin(rl) * right0 + np.cos(rl) * up0

        # The camera sits at aircraft_ref + offset (body frame right/up/forward). Move the
        # origin to the actual camera so the projection has the correct parallax.
        off_r, off_u, off_f = self.camera_offset_body
        offset_world = off_r * right + off_u * up_vec + off_f * forward

        # Boresight = airframe tilted by the fixed mount pitch (about the wing/right axis).
        mp = np.radians(self.camera_mount_pitch_deg)
        fwd_cam = np.cos(mp) * forward + np.sin(mp) * up_vec
        up_cam = -np.sin(mp) * forward + np.cos(mp) * up_vec
        right_cam = right

        v = np.array([east, north, up]) - offset_world      # vector from CAMERA to point
        return np.array([float(v.dot(right_cam)), float(v.dot(up_cam)), float(v.dot(fwd_cam))])

    def projectPointENU(self, north, east, up, camera_heading_deg, pitch_deg, roll_deg,
                        horizontal_fov, vertical_fov, screen_width, screen_height):
        """
        Project a point onto the image using a standard pinhole camera.

        This is the geometrically correct projection: the camera sits at the aircraft
        and looks along ``camera_heading_deg`` (which MUST be the heading the simulator
        actually renders, i.e. ``runway_heading - 180 + yaw``), pitched up by ``pitch_deg``
        and rolled by ``roll_deg`` about the boresight.

        Args:
            north, east, up: Vector from the camera (aircraft) to the target point, in
                metres, in a local ENU-style frame (north, east, up).
            camera_heading_deg: True heading the camera looks along, in degrees.
            pitch_deg: Aircraft pitch value from the metadata (MSFS convention), in degrees.
            roll_deg: Aircraft roll/bank value from the metadata, in degrees.
            horizontal_fov, vertical_fov: Field of view in degrees.
            screen_width, screen_height: Image size in pixels.

        Returns:
            Tuple (x_pixel, y_pixel, depth). ``depth`` is the distance along the boresight;
            it is > 0 when the point is in front of the camera (visible) and <= 0 behind it.

        NOTE: this projects a single point with no clipping; a point behind the camera
        (depth <= 0) yields a mirrored/garbage pixel. Annotation builders must go through
        ``visibleRunwayPolygon`` (near-plane + image-rectangle clip) instead of using this
        raw pixel for off-frame corners.

        Camera model (from the MSFS camera debug, see class constants):
          * MSFS PLANE_PITCH_DEGREES is positive NOSE-DOWN -> MSFS_PITCH_SIGN.
          * roll/yaw as-is (yaw is folded into camera_heading_deg upstream).
          * the camera is a rigid POSITION offset from the aircraft reference
            (self.camera_offset_body, body frame [right, up, forward]) ...
          * ... plus a fixed MOUNT PITCH of the boresight (self.camera_mount_pitch_deg).
        """
        x_c, y_c, z_c = self._cameraSpaceENU(north, east, up, camera_heading_deg,
                                             pitch_deg, roll_deg)
        return self._projectCameraSpace(np.array([x_c, y_c, z_c]), horizontal_fov,
                                        vertical_fov, screen_width, screen_height) + (z_c,)

    @staticmethod
    def _projectCameraSpace(pt_cam, horizontal_fov, vertical_fov, screen_width, screen_height):
        """Perspective-divide a camera-space point [x_c, y_c, z_c] to a (x_pixel, y_pixel)."""
        f_h = (screen_width / 2) / np.tan(np.radians(horizontal_fov) / 2)
        f_v = (screen_height / 2) / np.tan(np.radians(vertical_fov) / 2)
        x_c, y_c, z_c = float(pt_cam[0]), float(pt_cam[1]), float(pt_cam[2])
        if z_c == 0:
            z_c = 1e-9
        x_pixel = (screen_width / 2) + f_h * (x_c / z_c)
        y_pixel = (screen_height / 2) - f_v * (y_c / z_c)
        return (x_pixel, y_pixel)

    # ---------------------
    # Polygon clipping (Sutherland-Hodgman) - keeps off-frame runways correct
    # ---------------------

    @staticmethod
    def _clip_near_plane(corners_cam, z_near):
        """
        Clip an ordered camera-space polygon against the near plane ``z_c >= z_near``.

        Sutherland-Hodgman on the single half-space in front of the camera. Edges that
        cross the plane are split exactly on it, so every returned vertex has ``z_c >=
        z_near`` and can be perspective-divided safely (no behind-camera mirroring).

        Args:
            corners_cam: list of camera-space vertices (np.array [x_c, y_c, z_c]) in ring
                order.
            z_near: near clip depth in metres.

        Returns:
            list of np.array vertices (possibly empty if the polygon is entirely behind).
        """
        out = []
        n = len(corners_cam)
        for i in range(n):
            cur = corners_cam[i]
            prv = corners_cam[i - 1]
            cur_in = cur[2] >= z_near
            prv_in = prv[2] >= z_near
            if cur_in:
                if not prv_in:
                    out.append(RunwayTaggingEngine._lerp_to_z(prv, cur, z_near))
                out.append(cur)
            elif prv_in:
                out.append(RunwayTaggingEngine._lerp_to_z(prv, cur, z_near))
        return out

    @staticmethod
    def _lerp_to_z(p0, p1, z):
        """Point on segment p0->p1 whose depth (z component) equals ``z``."""
        t = (z - p0[2]) / (p1[2] - p0[2])
        return p0 + t * (p1 - p0)

    @staticmethod
    def _clip_rect(points_2d, width, height):
        """
        Clip a 2D polygon to the image rectangle [0,width] x [0,height].

        Sutherland-Hodgman against the four axis-aligned image edges (all convex).

        Args:
            points_2d: list of (x, y) pixel vertices in ring order.
            width, height: image size in pixels.

        Returns:
            list of (x, y) vertices of the clipped polygon (empty if fully outside).
        """
        # Each edge: (inside(x,y), intersect(a,b)) for the half-plane kept.
        def clip_edge(poly, inside, intersect):
            if not poly:
                return poly
            out = []
            n = len(poly)
            for i in range(n):
                cur = poly[i]
                prv = poly[i - 1]
                cur_in = inside(cur)
                prv_in = inside(prv)
                if cur_in:
                    if not prv_in:
                        out.append(intersect(prv, cur))
                    out.append(cur)
                elif prv_in:
                    out.append(intersect(prv, cur))
            return out

        def inter_x(a, b, xedge):
            t = (xedge - a[0]) / (b[0] - a[0])
            return (xedge, a[1] + t * (b[1] - a[1]))

        def inter_y(a, b, yedge):
            t = (yedge - a[1]) / (b[1] - a[1])
            return (a[0] + t * (b[0] - a[0]), yedge)

        poly = list(points_2d)
        poly = clip_edge(poly, lambda p: p[0] >= 0.0, lambda a, b: inter_x(a, b, 0.0))
        poly = clip_edge(poly, lambda p: p[0] <= width, lambda a, b: inter_x(a, b, width))
        poly = clip_edge(poly, lambda p: p[1] >= 0.0, lambda a, b: inter_y(a, b, 0.0))
        poly = clip_edge(poly, lambda p: p[1] <= height, lambda a, b: inter_y(a, b, height))
        return poly

    def visibleRunwayPolygon(self, corners_cam, horizontal_fov, vertical_fov,
                             screen_width, screen_height):
        """
        Return the runway polygon clipped to the visible image.

        Near-plane clip (drops/splits behind-camera vertices) -> perspective project ->
        image-rectangle clip. The result is the runway as it actually appears in frame.

        Args:
            corners_cam: the four runway corners in camera space (np.array [x_c, y_c, z_c]),
                in ring order A, B, C, D.
            horizontal_fov, vertical_fov: field of view in degrees.
            screen_width, screen_height: image size in pixels.

        Returns:
            list of (x, y) pixel vertices inside [0,W] x [0,H] (empty if the runway is
            entirely off-frame or behind the camera).
        """
        clipped_cam = self._clip_near_plane(corners_cam, self.NEAR_PLANE_M)
        if len(clipped_cam) < 3:
            return []
        projected = [self._projectCameraSpace(p, horizontal_fov, vertical_fov,
                                              screen_width, screen_height)
                     for p in clipped_cam]
        visible = self._clip_rect(projected, float(screen_width), float(screen_height))
        if len(visible) < 3:
            return []
        return visible

    def cornerPixelsFromMetadata(self, airport_data, generated_point, pitch, yaw, roll,
                                 horizontal_fov, vertical_fov, screen_width, screen_height):
        """
        Project the four runway corners to pixels using the correct pinhole camera.

        Rebuilds the runway corners from ``airport_data`` and projects them from the
        aircraft position (``generated_point`` = local north/east, z = height above the
        runway centre) with the camera facing ``runway_heading - 180 + yaw`` - the heading
        the simulator actually renders. Returns [(x, y) for A, B, C, D].
        """
        from tools.RunwayGeometryCalculator import RunwayGeometryCalculator

        rc = airport_data["runway_center"]
        center_alt = float(rc["altitude"])
        airport = RunwayGeometryCalculator(
            airport_data.get("name", ""), airport_data.get("icao_code", ""),
            airport_data.get("runway_name", ""),
            float(airport_data["runway_width"]), float(airport_data["runway_length"]),
            float(airport_data["runway_heading"]),
            float(rc["latitude"]), float(rc["longitude"]), center_alt,
            float(airport_data["start_height"]), float(airport_data["end_height"]), {})
        corners = airport.calculateRunwayCorners()

        gp = generated_point.tolist() if isinstance(generated_point, np.ndarray) else list(generated_point)
        ac_n, ac_e, ac_alt = gp[0], gp[1], center_alt + gp[2]
        camera_heading = float(airport_data["runway_heading"]) - 180.0 + yaw

        pixels = []
        for key in ["top_left", "top_right", "bottom_right", "bottom_left"]:
            cn, ce, ca = corners[key]
            x_pixel, y_pixel, _depth = self.projectPointENU(
                cn - ac_n, ce - ac_e, ca - ac_alt, camera_heading, pitch, roll,
                horizontal_fov, vertical_fov, screen_width, screen_height)
            pixels.append((int(round(x_pixel)), int(round(y_pixel))))
        return pixels

    def visiblePolygonFromMetadata(self, airport_data, generated_point, pitch, yaw, roll,
                                   horizontal_fov, vertical_fov, screen_width, screen_height):
        """
        Runway polygon clipped to the visible image, rebuilt from metadata.

        Same corner reconstruction as ``cornerPixelsFromMetadata`` (so both stay in sync),
        but returns the polygon clipped to the frame via ``visibleRunwayPolygon`` instead of
        four raw corner pixels. This is what the annotation builders use so off-frame and
        behind-camera corners never leak into ``segmentation``/``bbox``.

        Returns:
            list of (x, y) pixel vertices inside the image (empty if the runway is not in
            frame).
        """
        from tools.RunwayGeometryCalculator import RunwayGeometryCalculator

        rc = airport_data["runway_center"]
        center_alt = float(rc["altitude"])
        airport = RunwayGeometryCalculator(
            airport_data.get("name", ""), airport_data.get("icao_code", ""),
            airport_data.get("runway_name", ""),
            float(airport_data["runway_width"]), float(airport_data["runway_length"]),
            float(airport_data["runway_heading"]),
            float(rc["latitude"]), float(rc["longitude"]), center_alt,
            float(airport_data["start_height"]), float(airport_data["end_height"]), {})
        corners = airport.calculateRunwayCorners()

        gp = generated_point.tolist() if isinstance(generated_point, np.ndarray) else list(generated_point)
        ac_n, ac_e, ac_alt = gp[0], gp[1], center_alt + gp[2]
        camera_heading = float(airport_data["runway_heading"]) - 180.0 + yaw

        corners_cam = []
        for key in ["top_left", "top_right", "bottom_right", "bottom_left"]:
            cn, ce, ca = corners[key]
            corners_cam.append(self._cameraSpaceENU(
                cn - ac_n, ce - ac_e, ca - ac_alt, camera_heading, pitch, roll))
        return self.visibleRunwayPolygon(corners_cam, horizontal_fov, vertical_fov,
                                         screen_width, screen_height)

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
            # Correct projection: rebuild corners from metadata, project with the real
            # camera direction (runway_heading - 180 + yaw), and clip to the visible image
            # so off-frame / behind-camera corners never leak into the label.
            polygon = self.visiblePolygonFromMetadata(
                airport_data, generated_point, aircraft_orientation[0],
                aircraft_orientation[1], aircraft_orientation[2],
                horizontal_fov_degrees, vertical_fov_degrees, image_width, image_height)

            if not polygon:
                # Runway is entirely off-frame / behind the camera -> no annotation.
                print(f"[skip] runway not in frame: {screenshot_name}")
                continue

            pixel_coords = [int(round(c)) for xy in polygon for c in xy]

            # Calculate bounding box from the clipped (in-image) polygon
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
        
        # Draw the runway label for each structured object
        for obj in structured_objects:
            # Clipped visible polygon (exactly what the saved COCO segmentation contains),
            # so the overlay matches the annotation even when the runway runs off-frame.
            polygon = self.visiblePolygonFromMetadata(
                airport_data, generated_point, cam_pitch, cam_yaw, cam_roll,
                horizontal_fov_degrees, vertical_fov_degrees, screen_width, screen_height)

            if not polygon:
                continue

            poly_int = [(int(round(x)), int(round(y))) for (x, y) in polygon]

            # Draw the closed visible polygon.
            cv2.polylines(image, [np.array(poly_int, dtype=np.int32)], isClosed=True,
                          color=(0, 255, 0), thickness=1)

            # Label the four true runway corners (A,B,C,D) that fall inside the frame, for
            # reference alongside the clipped polygon.
            corner_px = self.cornerPixelsFromMetadata(
                airport_data, generated_point, cam_pitch, cam_yaw, cam_roll,
                horizontal_fov_degrees, vertical_fov_degrees, screen_width, screen_height)
            for label, (x_pixel, y_pixel) in zip(["A", "B", "C", "D"], corner_px):
                if 0 <= x_pixel < image.shape[1] and 0 <= y_pixel < image.shape[0]:
                    cv2.circle(image, (x_pixel, y_pixel), radius=3,
                              color=(0, 0, 255), thickness=-1)
                    cv2.putText(image, label, (x_pixel + 5, y_pixel - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save the annotated image
        if not cv2.imwrite(output_path, image):
            raise IOError(f"Failed to save image: {output_path}")

        print(f"Image saved: {output_path}")
