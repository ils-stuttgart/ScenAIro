import os
import json
import cv2
import numpy as np

from dependencies.SimConnect import SimConnect
from tools.AircraftPositioningAgent import AircraftPositioningAgent


class MetadataFileReader:
    """
    Liest eine ScenAIro/COCO-ähnliche JSON-Metadatendatei ein und kann daraus
    über den AircraftPositioningAgent wieder Screenshots + Overlays erzeugen.
    """

    def __init__(self, file_path, screenshot_dir=None):
        """
        :param file_path: Pfad zur JSON-Metadatendatei
        :param screenshot_dir: Basisverzeichnis für neue Screenshots.
                               Falls None: Verzeichnis der JSON-Datei.
        """
        self.file_path = file_path
        self.metadata = {}
        self.screenshot_dir = screenshot_dir or os.path.dirname(file_path)

    # ---------------------------------------------------------
    # Laden & Zugriff auf Metadaten
    # ---------------------------------------------------------

    def load_metadata(self):
        """Lädt die JSON-Metadaten von self.file_path."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Metadata JSON not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        return self.metadata

    def _ensure_loaded(self):
        """Hilfsfunktion: lädt Metadaten, falls noch nicht geschehen."""
        if not self.metadata:
            self.load_metadata()

    def get_image_info(self):
        """
        Gibt den ersten Eintrag aus dem 'images'-Block zurück.
        Erwartet ein COCO-ähnliches Format.
        """
        self._ensure_loaded()
        images = self.metadata.get("images", [])
        if not images:
            raise ValueError("Keine 'images'-Einträge in den Metadaten gefunden.")
        return images[0]

    def get_annotations(self):
        """Gibt die Liste der Annotationen zurück."""
        self._ensure_loaded()
        return self.metadata.get("annotations", [])

    # ---------------------------------------------------------
    # Hilfsfunktionen für Metadaten → Sim-Parameter
    # ---------------------------------------------------------

    def _get_aircraft_position(self):
        """
        Liest 'position_of_aircraft' aus der JSON.
        Erwartetes Format: [lat, lon, alt] (Strings oder Zahlen).
        """
        self._ensure_loaded()
        pos = self.metadata.get("position_of_aircraft")
        if pos is None:
            raise ValueError("Feld 'position_of_aircraft' fehlt in den Metadaten.")
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            lat, lon, alt = map(float, pos[:3])
            return lat, lon, alt
        raise ValueError(f"Ungültiges Format für 'position_of_aircraft': {pos}")

    def _get_runway_heading(self):
        """
        Liest 'runway_heading' aus 'runway_data'.
        """
        self._ensure_loaded()
        runway_data = self.metadata.get("runway_data") or self.metadata.get("runway", {})
        if not runway_data:
            raise ValueError("Feld 'runway_data' fehlt in den Metadaten.")
        heading = runway_data.get("runway_heading")
        if heading is None:
            raise ValueError("Feld 'runway_heading' fehlt in 'runway_data'.")
        return float(heading)

    def _get_daytime(self):
        """
        Liest 'daytime' aus der JSON.
        Erwartetes Format: { "hours": int, "minutes": int }
        """
        self._ensure_loaded()
        daytime = self.metadata.get("daytime", {})
        hours = int(daytime.get("hours", 12))
        minutes = int(daytime.get("minutes", 0))
        return hours, minutes

    # ---------------------------------------------------------
    # Bild aus JSON generieren (mit MSFS + AircraftPositioningAgent)
    # ---------------------------------------------------------

    def generate_image_from_metadata(
        self,
        use_sim=True,
        output_path=None,
        draw_bbox=True,
        draw_segmentation=True,
        line_thickness=2,
    ):
        """
        Erzeugt ein Bild auf Basis der JSON-Metadaten.

        Wenn use_sim=True:
            - Verwendet SimConnect + AircraftPositioningAgent, um das Flugzeug
              anhand der Metadaten neu zu positionieren und einen Screenshot zu machen.
        Wenn use_sim=False:
            - Verwendet nur ein leeres Canvas mit passender Auflösung.

        Anschließend werden Bounding Boxes und/oder Segmentierungen eingezeichnet.

        :param use_sim: Ob MSFS + AircraftPositioningAgent genutzt werden soll.
        :param output_path: Zielpfad für das erzeugte Bild.
                            Wenn None, wird `<json_basename>_from_json.png`
                            im gleichen Ordner wie die JSON gespeichert.
        :param draw_bbox:   True = BBox zeichnen.
        :param draw_segmentation: True = Segmentation-Polygone zeichnen.
        :param line_thickness: Linienstärke der Overlays.
        :return: Pfad zum erzeugten (ggf. getaggten) Bild.
        """
        self._ensure_loaded()

        # --- Bild-Infos aus JSON ---
        img_info = self.get_image_info()
        width = int(img_info.get("width", 2560))
        height = int(img_info.get("height", 1440))

        # Zielpfad bestimmen
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_path = os.path.join(self.screenshot_dir, f"{base_name}_from_json.png")

        # -----------------------------------------
        # Basisbild über MSFS erzeugen
        # -----------------------------------------
        if use_sim:
            try:
                lat, lon, alt = self._get_aircraft_position()
                runway_heading = self._get_runway_heading()
                hours, minutes = self._get_daytime()

                pitch = 0.0
                roll = 0.0
                # Wie im Data-Generator: Flugrichtung gegen Runway → Heading - 180
                heading = runway_heading - 180.0

                # SimConnect & AircraftPositioningAgent initialisieren
                sim = SimConnect()
                coord_setter = AircraftPositioningAgent(sim)

                # Conversion from meters to feet for msfs
                alt *= 3.28084
            

                # Screenshot-Pfad ist der Ordner der JSON
                screenshot_name = coord_setter.positionAircraftInSimAndTakeScreenshot(
                    lat,
                    lon,
                    alt,
                    pitch,
                    heading,
                    roll,
                    self.screenshot_dir,
                    width,
                    height,
                    hours,
                    minutes,
                    excludeImg=False  # wir wollen ja wirklich Bilder
                )

                base_image_path = os.path.join(self.screenshot_dir, f"{screenshot_name}.png")
                image = cv2.imread(base_image_path)
                if image is None:
                    print("[MetadataFileReader] Warnung: Screenshot konnte nicht geladen werden, "
                          "erstelle leeres Canvas.")
                    image = np.zeros((height, width, 3), dtype=np.uint8)
            except Exception as e:
                print(f"[MetadataFileReader] Fehler bei SimConnect/AircraftPositioningAgent: {e}")
                print("[MetadataFileReader] Fallback: leeres Canvas ohne Sim.")
                image = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Kein Sim: leeres Bild
            print("[MetadataFileReader] use_sim=False – erstelle leeres Canvas.")
            image = np.zeros((height, width, 3), dtype=np.uint8)

        # -----------------------------------------
        # Annotationen auf das Bild zeichnen
        # -----------------------------------------
        annotations = self.get_annotations()
        if not annotations:
            print("[MetadataFileReader] Warnung: Keine Annotationen im JSON gefunden.")

        # Farbe: BGR (Grün)
        color = (0, 255, 0)

        for anno in annotations:
            # BBox zeichnen
            if draw_bbox and "bbox" in anno:
                bbox = anno["bbox"]
                if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                    x, y, w, h = bbox
                    x1 = int(round(x))
                    y1 = int(round(y))
                    x2 = int(round(x + w))
                    y2 = int(round(y + h))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

            # Segmentation zeichnen (COCO-Style)
            if draw_segmentation and "segmentation" in anno:
                seg_list = anno["segmentation"]

                # Fall 1: [[x1,y1,x2,y2,...]]
                if isinstance(seg_list, list) and len(seg_list) > 0 and isinstance(seg_list[0], list):
                    segs = seg_list
                # Fall 2: [x1,y1,x2,y2,...] (flache Liste)
                elif isinstance(seg_list, list):
                    segs = [seg_list]
                else:
                    segs = []

                for seg in segs:
                    if not isinstance(seg, list) or len(seg) < 4:
                        continue
                    pts = np.array(
                        list(zip(seg[0::2], seg[1::2])),
                        dtype=np.int32
                    ).reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=line_thickness)

        # -----------------------------------------
        # Bild speichern
        # -----------------------------------------
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        success = cv2.imwrite(output_path, image)
        if not success:
            raise IOError(f"Fehler beim Speichern des Bildes unter: {output_path}")

        print(f"[MetadataFileReader] Bild aus JSON generiert: {output_path}")
        return output_path

    # ---------------------------------------------------------
    # Ordnerverarbeitung: Alle JSONs abarbeiten
    # ---------------------------------------------------------

    def process_folder(self, folder_path, use_sim=True):
        """
        Liest alle JSON-Dateien in einem Ordner und generiert jeweils ein Bild.
        :param folder_path: Pfad zum Ordner mit JSON-Dateien
        :param use_sim: Soll MSFS + AircraftPositioningAgent genutzt werden?
        :return: Liste der erzeugten Bilder
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Ordner nicht gefunden: {folder_path}")

        json_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".json")
        ]

        if not json_files:
            raise FileNotFoundError("Keine JSON-Dateien im Ordner gefunden.")

        print(f"[MetadataFileReader] Finde {len(json_files)} JSON-Dateien in {folder_path}")

        output_images = []

        for json_file in json_files:
            try:
                print(f"[MetadataFileReader] Verarbeite: {json_file}")

                reader = MetadataFileReader(json_file, screenshot_dir=folder_path)
                reader.load_metadata()

                out_path = reader.generate_image_from_metadata(
                    use_sim=use_sim
                )

                output_images.append(out_path)
                print(f"[MetadataFileReader] Fertig: {out_path}")

            except Exception as e:
                print(f"[MetadataFileReader] Fehler bei {json_file}: {e}")

        print(f"[MetadataFileReader] Alle Dateien abgearbeitet ({len(output_images)} Bilder erzeugt).")
        return output_images
