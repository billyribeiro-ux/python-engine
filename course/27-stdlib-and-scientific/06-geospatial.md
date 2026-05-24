# Geospatial

Geospatial work in Python is a triangle: **shapely** for shape geometry, **geopandas** for tabular spatial data, **pyproj** for projections. The big secret: most "GIS bugs" are coordinate-system confusion. Learn the right CRS and the rest is straightforward.

## CRS — Coordinate Reference Systems

Every set of coordinates exists in a CRS. The three you'll meet:

- **EPSG:4326** (WGS-84) — lat/lon in degrees. Used by GPS, JSON APIs, most web data.
- **EPSG:3857** (Web Mercator) — meters; the projection Google Maps uses. Distorted at high latitudes.
- **Local UTM zones** (EPSG:32601 to 32660 northern, 32701-32760 southern) — meters, accurate for one strip of longitude. Use for area / distance calculations.

The rule: **lat/lon for storage and APIs; UTM (or a local projection) for distance / area math**.

```python
import pyproj


# Distance between two lat/lon points (UK to Japan)
lon1, lat1 = -0.1, 51.5            # London
lon2, lat2 = 139.7, 35.7           # Tokyo

geod = pyproj.Geod(ellps="WGS84")
distance_m = geod.inv(lon1, lat1, lon2, lat2)[2]
print(f"{distance_m / 1000:.0f} km")    # ~9500 km
```

`pyproj.Geod` does great-circle distance on the WGS-84 ellipsoid. For most "real distance between two GPS points" needs, this is right.

## `shapely` — shape geometry

```python
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, box


p1 = Point(0, 0)
p2 = Point(3, 4)
print(p1.distance(p2))                  # 5.0 (Euclidean, in whatever units the coords are)


# Polygon from corners
poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
print(poly.area)                        # 1.0
print(poly.bounds)                      # (0.0, 0.0, 1.0, 1.0)
print(poly.centroid)                    # POINT (0.5 0.5)


# Predicates
print(poly.contains(Point(0.5, 0.5)))   # True
print(poly.intersects(box(0.5, 0.5, 2, 2)))   # True

# Operations
union = poly.union(box(0.5, 0.5, 2, 2))
diff = poly.difference(box(0.5, 0.5, 2, 2))
buffer = poly.buffer(0.5)               # grow by 0.5 in all directions
```

Shapely's coordinates are **unit-agnostic**. If your points are in WGS-84 (degrees), `.distance()` returns degrees — which is meaningless. **Project to a meter-based CRS first** for any distance / area computation.

## `geopandas` — spatial tabular data

```python
import geopandas as gpd
from shapely.geometry import Point


# Build a GeoDataFrame from records
data = [
    {"city": "London", "country": "UK", "geometry": Point(-0.1, 51.5)},
    {"city": "Tokyo", "country": "JP", "geometry": Point(139.7, 35.7)},
    {"city": "Sydney", "country": "AU", "geometry": Point(151.2, -33.9)},
]
gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")


# Project to a metric CRS
gdf_m = gdf.to_crs("EPSG:3857")            # web mercator (rough)
# OR: pick the right UTM per region for accurate distances


# Read from a file (shapefile, GeoJSON, GeoPackage)
counties = gpd.read_file("counties.geojson")


# Spatial join: find which county each city is in
joined = gpd.sjoin(gdf, counties, predicate="within")
```

`gpd.sjoin` is the killer feature — joining a points dataframe to a polygons dataframe based on geometric containment. The pandas equivalent of "WHERE point IS INSIDE polygon."

## Reading common formats

| Format | Read |
|---|---|
| GeoJSON | `gpd.read_file("data.geojson")` |
| Shapefile | `gpd.read_file("data.shp")` |
| GeoPackage | `gpd.read_file("data.gpkg")` |
| KML | `gpd.read_file("data.kml")` |
| PostGIS | `gpd.read_postgis("SELECT * FROM table", engine)` |

`fiona` (used by geopandas) handles dozens of formats. For new data, prefer GeoPackage (modern, self-contained, one file).

## A worked example: which ZIP codes are within 10km of a point?

```python
import geopandas as gpd
from shapely.geometry import Point


# Load ZIP code polygons (assume WGS-84)
zips = gpd.read_file("us_zips.geojson")

# Our point of interest
center = Point(-122.4194, 37.7749)          # San Francisco
center_gdf = gpd.GeoDataFrame([{"geometry": center}], crs="EPSG:4326")

# Project to a meter-based CRS appropriate for the region
center_m = center_gdf.to_crs(epsg=3857)
zips_m = zips.to_crs(epsg=3857)

# Buffer 10 km
buffer = center_m.geometry.iloc[0].buffer(10_000)

# Which zips intersect?
nearby = zips_m[zips_m.intersects(buffer)]
print(f"{len(nearby)} zips within 10km")
```

For more accuracy than web mercator at this scale, project to UTM zone 10N (`epsg=32610` for San Francisco).

## Reverse geocoding without an API

```python
import geopandas as gpd
from shapely.geometry import Point


def reverse_geocode(lat: float, lon: float, polygons: gpd.GeoDataFrame, name_col: str = "name"):
    point = Point(lon, lat)
    match = polygons[polygons.geometry.contains(point)]
    if match.empty:
        return None
    return match.iloc[0][name_col]
```

Load country / region polygons (Natural Earth has free ones); call this with a coordinate; get the country / region name. Faster than calling an API for millions of points.

## Distance calculations

Three options, in increasing accuracy:

```python
import math
import pyproj
from shapely.geometry import Point


lat1, lon1 = 51.5, -0.1
lat2, lon2 = 35.7, 139.7


# 1. Haversine — simple, ~99% accurate for most uses
def haversine(lat1, lon1, lat2, lon2, R=6_371_000):
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# 2. Geodesic — accurate (WGS-84 ellipsoid)
geod = pyproj.Geod(ellps="WGS84")
distance_m = geod.inv(lon1, lat1, lon2, lat2)[2]


# 3. Via projection — for arbitrary geometry distances
import geopandas as gpd
gdf = gpd.GeoDataFrame(
    [{"geometry": Point(lon1, lat1)}, {"geometry": Point(lon2, lat2)}],
    crs="EPSG:4326",
).to_crs("ESRI:54032")          # World Azimuthal Equidistant
print(gdf.geometry.iloc[0].distance(gdf.geometry.iloc[1]))
```

For most apps, haversine. For finance / surveying / aviation, geodesic.

## Visualisation

```python
import matplotlib.pyplot as plt


fig, ax = plt.subplots(figsize=(10, 6))
gdf.plot(ax=ax, color="red", markersize=50)
nearby_zips.plot(ax=ax, color="lightblue", edgecolor="black", alpha=0.5)
plt.show()
```

For interactive maps: `folium` (Leaflet wrapper) or `plotly.express.scatter_mapbox`.

## Pitfalls

!!! warning "Distance in degrees"
    `point1.distance(point2)` in WGS-84 returns *degrees*, not meters. A degree of longitude at the equator is ~111km; at 80°N it's ~19km. Always project before distance.

!!! warning "Polygon orientation"
    Shapely polygons should be counter-clockwise (exterior ring). Reversed orientation can cause `contains()` to misbehave on edge cases.

!!! warning "PostGIS vs GeoPandas geometry"
    PostGIS expects WKT or WKB; geopandas wraps it. When inserting geopandas data into PostGIS, use `gdf.to_postgis(...)`.

!!! warning "Loading big shapefiles**
    A national-level boundary shapefile is hundreds of MB. Load once, save as parquet (`gdf.to_parquet`) for fast re-reads.

## Bottom line

For geospatial:

- **Lat/lon (EPSG:4326)** for storage.
- **Project to UTM / appropriate metric CRS** for distance and area.
- **`shapely`** for shape geometry; **`geopandas`** for tables of shapes; **`pyproj`** for projections.
- **`gpd.sjoin`** for spatial joins.
- **Haversine** is good enough for most distance work; **geodesic** for precision.

Continue to **[SciPy — signal, optimize, ndimage](07-scipy.md)**.
