import math
import json

class RunwayGeometryCalculator:

    def __init__(self, name, icao_code, runway_name, runway_width, runway_length, runway_heading, 
                 center_lat, center_long, center_alt, start_height, end_height, runway_attributes):    
        self.name = name
        self.icao_code = icao_code
        self.runway_name = runway_name
        self.runway_width = float(runway_width)
        self.runway_length = float(runway_length)
        self.runway_heading = float(runway_heading)
        self.runway_center = {"latitude": center_lat, "longitude": center_long, "altitude": center_alt}
        self.start_height = start_height
        self.end_height = end_height
        self.runway_attributes = runway_attributes

    def calculateRunwayCorners(self):
        """Berechnet die kartesischen Koordinaten der Landebahn-Eckpunkte."""
        heading_rad = math.radians(self.runway_heading)
        runwayAltitude = self.runway_center["altitude"]
        half_length = self.runway_length / 2
        half_width = self.runway_width / 2
        start_height = self.start_height
        end_height = self.end_height
        startCornerHeight = runwayAltitude + (start_height - runwayAltitude)
        endCornerHeight = runwayAltitude + (end_height - runwayAltitude)
        corners = {
            "top_left": (*self.alignCornersWithRunwayHeading(-half_length, half_width, heading_rad), endCornerHeight),
            "top_right": (*self.alignCornersWithRunwayHeading(-half_length, -half_width, heading_rad), endCornerHeight),
            "bottom_left": (*self.alignCornersWithRunwayHeading(half_length, half_width, heading_rad), startCornerHeight),
            "bottom_right": (*self.alignCornersWithRunwayHeading(half_length, -half_width, heading_rad), startCornerHeight),
        }
        return corners

    def alignCornersWithRunwayHeading(self, x, y, angle_rad):
        """Rotiert einen Punkt um den Ursprung basierend auf dem Heading-Winkel."""
        x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        return round(x_rot, 2), round(y_rot, 2)

    def createAirport(self):
        return {
            "airport_name": self.name,
            "icao_code": self.icao_code,
            "runway": {
                "name": self.runway_name,
                "width": self.runway_width,
                "length": self.runway_length,
                "heading": self.runway_heading,
                "center_coordinates": self.runway_center,
            },
            "start_height": self.start_height,
            "end_height": self.end_height,
            "attributes": self.runway_attributes,
        }

    @classmethod
    def createAirportConfig(cls, data):
        """Erstellt ein Airport-Objekt aus einem Dictionary."""
        print("Creating RunwayCalc from dict")
        runway = data["runway"]
        center_coords = runway["center_coordinates"]
        return cls(
            name=data["airport_name"],
            icao_code=data["icao_code"],
            runway_name=runway["name"],
            runway_width=runway["width"],
            runway_length=runway["length"],
            runway_heading=runway["heading"],
            center_lat=center_coords["latitude"],
            center_long=center_coords["longitude"],
            center_alt=center_coords["altitude"],
            start_height=runway["start_height"],
            end_height=runway["end_height"],
            runway_attributes=runway.get("attributes", {})
        )

    def saveAirport(self, filename):
        """Speichert die Flughafendaten in eine JSON-Datei."""
        print(f"Saving RunwayCalc to file: {filename}")
        with open(filename, "w") as file:
            json.dump(self.createAirport(), file, indent=4)

    @classmethod
    def loadAirport(cls, filename):
        """Lädt ein Flughafen-Objekt aus einer JSON-Datei."""
        print(f"Loading RunwayCalc from file: {filename}")
        with open(filename, "r") as file:
            data = json.load(file)
        return cls.createAirportConfig(data)
