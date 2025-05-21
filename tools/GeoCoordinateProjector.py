import numpy as np
from pyproj import Geod

class GeoCoordinateProjector:
    geod = Geod(ellps='WGS84')

    @staticmethod
    def transform_points(points, center_lat, center_lon, center_alt, heading):
        """
        Transformiert kartesische Punkte (x, y, z) in geografische Koordinaten (Breitengrad, Längengrad, Höhe).

        :param points: Liste von Punkten (x, y, z) in kartesischen Koordinaten.
        :param center_lat: Breitengrad des Mittelpunktes.
        :param center_lon: Längengrad des Mittelpunktes.
        :param center_alt: Höhe des Mittelpunktes.
        :param heading: Kurs in Grad.
        :return: Liste der transformierten Punkte in (Breitengrad, Längengrad, Höhe).
        """
        heading_rad = np.radians(heading)
        transformed_points = []

        for x, y, z in points:
            distance = np.sqrt(x ** 2 + y ** 2)
            azimuth = np.degrees(heading_rad + np.arctan2(y, x))
            lon_new, lat_new, back_azimuth = GeoCoordinateProjector.geod.fwd(center_lon, center_lat, azimuth, distance)
            alt_new = center_alt + z
            transformed_points.append((lat_new, lon_new, alt_new))

        return transformed_points
