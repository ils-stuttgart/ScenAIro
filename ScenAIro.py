import tkinter as tk
from tkinter import messagebox
import numpy as np

from dependencies.SimConnect import *

from .presentation.ScenAIroUI import ScenAIroUI
from .tools.AutomatedRunwayTaggingCopy import AutomatedRunwayTagging
from .tools.PointCloudGenerator import PointCloudGenerator
from .tools.RunwayCalc import RunwayCalc
from .tools.ConeTransformer import ConeTransformer
from .tools.CoordSetter import CoordSetter
from .tools.RunwayCornerAnnotationStruct import RunwayCornerAnnotationStruct

class ScenAIro(tk.Tk):
    airport = RunwayCalc()

    def __init__(self):
        super().__init__()
        # Fenster-Einstellungen
        self.title("ScenAIro")
        self.geometry("1400x820")
        self.configure(bg="#f0f4f8")

        # Initialisierung von Attributen
        self.airport = None
        self.points = None
        self.geo_points = None
        self.angles = None
        self.tagging = AutomatedRunwayTagging()  # Instanz von AutomatedRunwayTagging
        self.coordsetter = CoordSetter()
        self.runwayCornerAnnotation = RunwayCornerAnnotationStruct()
        self.pointCloudGeneration = PointCloudGenerator()
        self.transformCone = ConeTransformer()

        # UI-Elemente erstellen
        self.ui = ScenAIroUI(self)

    def fill_fields(self, entry_fields, values):
        for key, value in values.items():
            entry_fields[key].delete(0, tk.END)
            entry_fields[key].insert(0, value)

    def validate_float(self, value, field_name):
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Invalid value for {field_name}: {value}")

    def validate_int(self, value, field_name):
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid value for {field_name}: {value}")

    def generate_and_transform_points(self):
        try:
            # Eingabewerte aus der GUI abrufen
            name = self.ui.airport_entries["Airport Name"].get()
            icao = self.ui.airport_entries["ICAO Code"].get()
            runway_name = self.ui.airport_entries["Runway Name"].get()
            width = float(self.ui.airport_entries["Width"].get())
            length = float(self.ui.airport_entries["Length"].get())
            runwayHeading = float(self.ui.airport_entries["Heading"].get())
            latitude = float(self.ui.airport_entries["Latitude"].get())
            longitude = float(self.ui.airport_entries["Longitude"].get())
            altitude = float(self.ui.airport_entries["Altitude"].get())
            start_height = float(self.ui.airport_entries["Start Height"].get())
            end_height = float(self.ui.airport_entries["End Height"].get())

            # Validierung der Eingabedaten
            if not all([name, icao, runway_name]):
                raise ValueError("Die Felder 'Airport Name', 'ICAO Code' und 'Runway Name' dürfen nicht leer sein.")

            # `RunwayCalc`-Objekt erstellen
            self.airport = RunwayCalc(
                name, icao, runway_name, width, length, runwayHeading, latitude, longitude, altitude, start_height, end_height, {}
            )

            # Punktgenerierungsparameter auslesen
            apex_x = self.validate_float(self.ui.point_entries["Apex X"].get(), "Apex X")
            apex_y = self.validate_float(self.ui.point_entries["Apex Y"].get(), "Apex Y")
            apex_z = self.validate_float(self.ui.point_entries["Apex Z"].get(), "Apex Z")
            self.apex = (apex_x, apex_y, apex_z)  # Speichere Apex als Instanzvariable

            self.lateral_angle_left = self.validate_float(self.ui.point_entries["Lateral Angle Left"].get(), "Lateral Angle Left")
            self.lateral_angle_right = self.validate_float(self.ui.point_entries["Lateral Angle Right"].get(), "Lateral Angle Right")
            self.vertical_min_angle = self.validate_float(self.ui.point_entries["Vertical Min Angle"].get(), "Vertical Min Angle")
            self.vertical_max_angle = self.validate_float(self.ui.point_entries["Vertical Max Angle"].get(), "Vertical Max Angle")
            self.max_distance = self.validate_float(self.ui.point_entries["Maximum Distance"].get(), "Maximum Distance")
            num_points = self.validate_int(self.ui.point_entries["Number of Points"].get(), "Number of Points") 

            self.points, self.apex_transformed = PointCloudGenerator.generate_cone(
                apex=self.apex,
                lateral_angle_left=self.lateral_angle_left,
                lateral_angle_right=self.lateral_angle_right,
                vertical_min_angle=self.vertical_min_angle,
                vertical_max_angle=self.vertical_max_angle,
                max_distance=self.max_distance,
                num_points=num_points,
                heading=self.airport.runway_heading
            )

            # Plot aktualisieren
            self.ui.update_plot(self.points, self.airport, self.apex_transformed)

        except ValueError as ve:
            messagebox.showerror("Error", f"Invalid input: {ve}")
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")

        return self.points

    def calculate_vertical_fov(self, horizontal_fov_degrees, aspect_ratio):
        horizontal_fov_radians = np.radians(horizontal_fov_degrees)
        vertical_fov_radians = 2 * np.arctan(np.tan(horizontal_fov_radians / 2) / aspect_ratio)
        print (f"Vert FOV {np.degrees(vertical_fov_radians)}")
        return np.degrees(vertical_fov_radians)
    
    def calculate_horizontal_fov(self, vertical_fov_degrees, aspect_ratio):
        vertical_fov_radians = np.radians(vertical_fov_degrees)
        horizontal_fov_radians = 2 * np.arctan(np.tan(vertical_fov_radians / 2) * aspect_ratio)
        return np.degrees(horizontal_fov_radians)

    def create_data(self):
        if self.ui.labeling_var.get():
            try:
                if hasattr(self, "_creating_data"):  # Schutz gegen Rekursion
                    raise RuntimeError("create_data wird bereits ausgeführt!")

                self._creating_data = True  # Markiere die Funktion als aktiv
                messagebox.showinfo("Create Data", "Creating labeled data...")

                # Punkte in Geo-Koordinaten generieren und transformieren
                generated_points = self.generate_and_transform_points()
                if generated_points is None or len(generated_points) == 0:
                    raise ValueError("Keine generierten Punkte verfügbar. Bitte Eingabewerte überprüfen.")

                # Flughafenzentrum aus runway_center abrufen
                center_lat = self.airport.runway_center["latitude"]
                center_lon = self.airport.runway_center["longitude"]
                center_alt = self.airport.runway_center["altitude"]
                heading = 0

                # Transformiere Punkte in Geo-Koordinaten
                # Heading Wert entfernen, da dieser nicht benötigt wird
                geo_points = ConeTransformer.transform_points(
                    points=generated_points,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    center_alt=center_alt,
                    heading=heading
                )

                if geo_points is None or len(geo_points) == 0:
                    raise ValueError("Die Transformation der Punkte in Geo-Koordinaten ist fehlgeschlagen.")

                # Screenshot-Pfad
                screenshot_path = r'C:\Users\MSFS2020_AI\Desktop\Saymon\images\TrainingData\NoRunway_ClearSkies\1800'

                # Berechnung der Runway-Eckpunkte
                corners = self.airport.calculate_runway_corners()

                # Start SimConnect
                sim = SimConnect()
                coord_setter = CoordSetter(sim)

                # Punkte verarbeiten und markieren
                for i, (geo_point, generated_point) in enumerate(zip(geo_points[1:], generated_points)):
                    latitude, longitude, altitude = map(float, geo_point)
                    altitude *= 3.28084  # Umrechnung in Fuß
                    pitch = 0.0
                    roll = 0.0
                    vertical_fov_degrees = np.degrees(1.028) 
                    screen_width = 3840
                    screen_height = 2160
                    aspectRatio = screen_width / screen_height
                    runwayHeading = self.airport.runway_heading
                    horizontal_fov_degrees = self.calculate_horizontal_fov(vertical_fov_degrees, aspectRatio)


                    screenshot_name = coord_setter.set_aircraft_values_and_screenshot(
                        latitude, longitude, altitude, pitch, (runwayHeading-180), roll, screenshot_path, screen_width, screen_height
                    )
    	            

                    runway_annotation = RunwayCornerAnnotationStruct()
                    structured_objects = runway_annotation.create_structured_objects(generated_point, corners, (0, 0, 0), runwayHeading, center_alt)

                    # Alle Rechtecke auf jedes Bild zeichnen
                    self.tagging.draw_points_on_existing_image(
                        image_path=f"{screenshot_path}\\{screenshot_name}.png",
                        output_path=f"{screenshot_path}\\tagged_{screenshot_name}.png",
                        structured_objects=structured_objects,
                        horizontal_fov_degrees=horizontal_fov_degrees,
                        vertical_fov_degrees=vertical_fov_degrees,
                        screen_width=screen_width,
                        screen_height=screen_height,
                        cam_pitch=0,
                        cam_yaw=0,
                        cam_roll=0
                    )


                messagebox.showinfo("Success", "Daten und Screenshots erfolgreich erstellt.")
                return None
            except Exception as e:
                messagebox.showerror("Error", f"Ein Fehler ist aufgetreten: {e}")
        else:
            messagebox.showinfo("Create Data", "Daten werden ohne Labeling erstellt...")
        