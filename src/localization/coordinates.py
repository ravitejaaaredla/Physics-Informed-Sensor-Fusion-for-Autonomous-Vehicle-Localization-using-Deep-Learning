import math


EARTH_RADIUS = 6378137.0


def xy_to_latlon(x, y, lat0, lon0):
    """
    Convert local x,y meters to latitude and longitude.

    x = east/west local movement in meters
    y = north/south local movement in meters
    """

    lat0_rad = math.radians(lat0)
    lon0_rad = math.radians(lon0)

    lat_rad = lat0_rad + (y / EARTH_RADIUS)
    lon_rad = lon0_rad + (x / (EARTH_RADIUS * math.cos(lat0_rad)))

    lat = math.degrees(lat_rad)
    lon = math.degrees(lon_rad)

    return lat, lon
