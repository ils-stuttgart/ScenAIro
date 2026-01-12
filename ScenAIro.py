import tkinter as tk
from tkinter import messagebox
import numpy as np

from dependencies.SimConnect import *

from presentation.ScenAIroUI import ScenAIroUI
from tools.RunwayTaggingEngine import RunwayTaggingEngine
from tools.SamplingPointGenerator import SamplingPointGenerator
from tools.RunwayGeometryCalculator import RunwayGeometryCalculator
from tools.GeoCoordinateProjector import GeoCoordinateProjector
from tools.AircraftPositioningAgent import AircraftPositioningAgent
from tools.RunwayCornerAnnotationStruct import RunwayCornerAnnotationStruct

class ScenAIro(tk.Tk):
    
    #airport = RunwayGeometryCalculator()

    def __init__(self):
        super().__init__()
        # Window Settings
        self.title("ScenAIro")
        self.geometry("1400x820")
        self.configure(bg="#f0f4f8")

        # Initialize Attributes
        sim = SimConnect()

        self.airport = None
        self.points = None
        self.geo_points = None
        self.angles = None
        self.tagging = RunwayTaggingEngine()  
        self.coordsetter = AircraftPositioningAgent(sim) 
        self.runwayCornerAnnotation = RunwayCornerAnnotationStruct()
        self.pointCloudGeneration = SamplingPointGenerator()
        self.transformCone = GeoCoordinateProjector()


        # create UI-element
        self.ui = ScenAIroUI(self)
        self.generateSampleDataset(silent=True)
        

    def populateDefaultParameters(self, entry_fields, values):
        for key, value in values.items():
            entry_fields[key].delete(0, tk.END)
            entry_fields[key].insert(0, value)

    def __isFloatValue(self, value, field_name):
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"Invalid value for {field_name}: {value}")

    def __isIntValue(self, value, field_name):
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid value for {field_name}: {value}")

    def generateSampleDataset(self, silent=False):
        try:
            # Get input values from UI

            try:
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
            except ValueError:
                raise ValueError("Please fill all fields correctly")

            # Validate Input Data
            if not all([name, icao, runway_name]):
                raise ValueError("Fields 'Airport Name', 'ICAO Code' and 'Runway Name' can't be empty")

            # create `RunwayCalc`- object
            self.airport = RunwayGeometryCalculator(
                name, icao, runway_name, width, length, runwayHeading, latitude, longitude, altitude, start_height, end_height, {}
            )

            # Punktgenerierungsparameter auslesen
            apex_x = self.__isFloatValue(self.ui.point_entries["Apex X"].get(), "Apex X")
            apex_y = self.__isFloatValue(self.ui.point_entries["Apex Y"].get(), "Apex Y")
            apex_z = self.__isFloatValue(self.ui.point_entries["Apex Z"].get(), "Apex Z")
            self.apex = (apex_x, apex_y, apex_z) 

            self.lateral_angle_left = self.__isFloatValue(self.ui.point_entries["Lateral Angle Left"].get(), "Lateral Angle Left")
            self.lateral_angle_right = self.__isFloatValue(self.ui.point_entries["Lateral Angle Right"].get(), "Lateral Angle Right")
            self.vertical_min_angle = self.__isFloatValue(self.ui.point_entries["Vertical Min Angle"].get(), "Vertical Min Angle")
            self.vertical_max_angle = self.__isFloatValue(self.ui.point_entries["Vertical Max Angle"].get(), "Vertical Max Angle")
            self.max_distance = self.__isFloatValue(self.ui.point_entries["Maximum Distance"].get(), "Maximum Distance")
            num_points = self.__isIntValue(self.ui.point_entries["Number of Points"].get(), "Number of Points")

            self.pitchMinAngle = self.__isFloatValue(self.ui.angle_entries["Pitch Min"].get(), "Pitch Min")
            self.pitchMaxAngle = self.__isFloatValue(self.ui.angle_entries["Pitch Max"].get(), "Pitch Max")
            self.yawMinAngle = self.__isFloatValue(self.ui.angle_entries["Yaw Min"].get(), "Yaw Min")
            self.yawMaxAngle = self.__isFloatValue(self.ui.angle_entries["Yaw Max"].get(), "Yaw Max")
            self.rollMinAngle = self.__isFloatValue(self.ui.angle_entries["Bank Min"].get(), "Bank Min")
            self.rollMaxAngle = self.__isFloatValue(self.ui.angle_entries["Bank Max"].get(), "Bank Max")

            aircraftOrientationAngles = {
                "pitchMin": self.pitchMinAngle,
                "pitchMax": self.pitchMaxAngle,
                "yawMin": self.yawMinAngle,
                "yawMax": self.yawMaxAngle,
                "rollMin": self.rollMinAngle,
                "rollMax": self.rollMaxAngle
            }
            
            # Distribution Settings
            self.distribution_settings = {
                "type": self.ui.distribution_var.get(), 
                "apply_x": self.ui.apply_x.get(),       
                "apply_y": self.ui.apply_y.get()        
            }

            self.points, self.apex_transformed, self.aircraftOrientation = (
                SamplingPointGenerator.generateCone(
                    apex=self.apex,
                    lateral_angle_left=self.lateral_angle_left,
                    lateral_angle_right=self.lateral_angle_right,
                    vertical_min_angle=self.vertical_min_angle,
                    vertical_max_angle=self.vertical_max_angle,
                    max_distance=self.max_distance,
                    num_points=num_points,
                    heading=self.airport.runway_heading,
                    aircraftOrientationAngles=aircraftOrientationAngles,
                    distribution_settings=self.distribution_settings 
                )
            )

            # Plot aktualisieren
            self.ui.refreshPlot(self.points, self.airport, self.apex_transformed)
            
            # Wenn erfolgreich, gib Punkte zurück
            return self.points

        except ValueError as ve:
            # WICHTIG: Nur meckern, wenn NICHT silent
            if not silent:
                messagebox.showerror("Error", f"Invalid input: {ve}")
            else:
                # Im Silent-Modus (beim Tippen) einfach nichts tun (oder in Konsole loggen)
                print(f"[Preview] Wartet auf valide Eingabe... ({ve})")
            return None
            
        except Exception as e:
            # Unerwartete Fehler zeigen wir trotzdem an, oder loggen sie zumindest laut
            print(f"[Error] Unexpected: {e}")
            if not silent:
                messagebox.showerror("Error", f"A really unexpected error occurred: {e}")
            return None

    def __calculateVerticalFOV(self, horizontal_fov_degrees, aspect_ratio):
        horizontal_fov_radians = np.radians(horizontal_fov_degrees)
        vertical_fov_radians = 2 * np.arctan(np.tan(horizontal_fov_radians / 2) / aspect_ratio)
        print (f"Vert FOV {np.degrees(vertical_fov_radians)}")
        return np.degrees(vertical_fov_radians)
    
    def __calculateHorizontalFOV(self, vertical_fov_degrees, aspect_ratio):
        vertical_fov_radians = np.radians(vertical_fov_degrees)
        horizontal_fov_radians = 2 * np.arctan(np.tan(vertical_fov_radians / 2) * aspect_ratio)
        return np.degrees(horizontal_fov_radians)

    def generateData(self):
        # Prüfung, ob bereits ein Prozess läuft
        if hasattr(self, "_creating_data"):
            messagebox.showwarning("Busy", "Data generation is already running!")
            return

        # Markierung setzen: Prozess läuft
        self._creating_data = True
        
        try:
            if self.ui.labeling_var.get():
                messagebox.showinfo("Create Data", "Creating labeled data...")

                # --- 1. Punkte generieren ---
                generated_points = self.generateSampleDataset(silent=False) # Silent=False, damit Fehler angezeigt werden
                if generated_points is None or len(generated_points) == 0:
                    raise ValueError("Keine generierten Punkte verfügbar. Bitte Eingabewerte überprüfen.")

                # --- 2. Metadaten vorbereiten ---
                center_lat = self.airport.runway_center["latitude"]
                center_lon = self.airport.runway_center["longitude"]
                center_alt = self.airport.runway_center["altitude"]
                heading = 0 

                # --- 3. Transformation in Geo-Koordinaten ---
                geo_points = GeoCoordinateProjector.transform_points(
                    points=generated_points,
                    center_lat=center_lat,
                    center_lon=center_lon,
                    center_alt=center_alt,
                    heading=heading
                )

                if geo_points is None or len(geo_points) == 0:
                    raise ValueError("Die Transformation der Punkte in Geo-Koordinaten ist fehlgeschlagen.")

                # Pfade und Einstellungen
                screenshot_path = r'C:\Users\mfs2024\Desktop\Saymon\Test' 
                
                # Checkboxen auslesen
                excludeImg = self.ui.labeling_exclImg.get()         
                createOverlay = self.ui.validation_var.get()        

                corners = self.airport.calculateRunwayCorners()
                setSimHour = self.__isIntValue(self.ui.time_entries["Hours"].get(), "Hours")
                setSimMin  = self.__isIntValue(self.ui.time_entries["Minutes"].get(), "Minutes")

                airport_metadata = {
                    "name": self.airport.name,
                    "icao_code": self.airport.icao_code,
                    "runway_name": self.airport.runway_name,
                    "runway_width": self.airport.runway_width,
                    "runway_length": self.airport.runway_length,
                    "runway_heading": self.airport.runway_heading,
                    "runway_center": self.airport.runway_center,
                    "start_height": self.airport.start_height,
                    "end_height": self.airport.end_height,
                }
                
                dist_settings = getattr(self, "distribution_settings", {"type": "Uniform", "apply_x": False, "apply_y": False})

                cone_metadata = {
                    "apex": self.apex,
                    "lateral_angle_left": self.lateral_angle_left,
                    "lateral_angle_right": self.lateral_angle_right,
                    "vertical_min_angle": self.vertical_min_angle,
                    "vertical_max_angle": self.vertical_max_angle,
                    "max_distance": self.max_distance,
                    "number_of_points": len(generated_points),
                    "distribution": dist_settings
                }

                daytime = {"hours": setSimHour, "minutes": setSimMin}

                # --- 4. SimConnect & Loop ---
                sim = SimConnect()
                coord_setter = AircraftPositioningAgent(sim)

                for i, (geo_point, generated_point) in enumerate(zip(geo_points[1:], generated_points)):
                    latitude, longitude, altitude = map(float, geo_point)
                    altitude *= 3.28084  # Meter -> Fuß

                    pitch = float(self.aircraftOrientation["pitch"][i])
                    roll = float(self.aircraftOrientation["roll"][i])
                    yaw_offset = float(self.aircraftOrientation["yaw"][i])

                    vertical_fov_degrees = np.degrees(0.8)
                    screen_width = 2560
                    screen_height = 1440
                    aspectRatio = screen_width / screen_height
                    runwayHeading = self.airport.runway_heading
                    horizontal_fov_degrees = self.__calculateHorizontalFOV(vertical_fov_degrees, aspectRatio)
                    
                    final_heading = (runwayHeading - 180) + yaw_offset

                    # Screenshot erstellen (nur wenn excludeImg False ist)
                    screenshot_name = coord_setter.positionAircraftInSimAndTakeScreenshot(
                        latitude, longitude, altitude, pitch, final_heading, roll, 
                        screenshot_path, screen_width, screen_height, setSimHour, setSimMin, excludeImg,
                    )
                    
                    runway_annotation = RunwayCornerAnnotationStruct()
                    structured_objects = runway_annotation.calculateAirplane2RunwayCornerStructure(
                        generated_point, corners, (0, 0, 0), runwayHeading, center_alt
                    )

                    # --- LOGIK FÜR OVERLAY ---
                    # Wir überspringen das Zeichnen (excludeImg=True an tagging engine übergeben), wenn:
                    # 1. Wir generell keine Bilder wollen (excludeImg)
                    # 2. ODER wenn wir explizit keine Overlays wollen (not createOverlay)
                    skip_tagging_image = excludeImg or not createOverlay

                    self.tagging.doOverlayLabelsOnImage(
                        image_path=f"{screenshot_path}\\{screenshot_name}.png",
                        output_path=f"{screenshot_path}\\tagged_{screenshot_name}.png",
                        structured_objects=structured_objects,
                        horizontal_fov_degrees=horizontal_fov_degrees,
                        vertical_fov_degrees=vertical_fov_degrees,
                        screen_width=screen_width,
                        screen_height=screen_height,
                        cam_pitch= -pitch,
                        cam_yaw= -yaw_offset,
                        cam_roll= -roll,
                        airport_data = airport_metadata,
                        cone_data= cone_metadata, 
                        geo_point= geo_point,
                        generated_point= generated_point,
                        daytime=daytime,
                        
                        # HIER IST DIE ÄNDERUNG:
                        excludeImg= skip_tagging_image 
                    )

                messagebox.showinfo("Success", "Daten und Screenshots erfolgreich erstellt.")

            else:
                messagebox.showinfo("Create Data", "Daten werden ohne Labeling erstellt...")

        except Exception as e:
            messagebox.showerror("Error", f"Ein Fehler ist aufgetreten: {e}")
            print(e)

        finally:
            if hasattr(self, "_creating_data"):
                del self._creating_data
        

