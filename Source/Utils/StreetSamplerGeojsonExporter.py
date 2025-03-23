import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

class StreetSamplerGeojsonExporter:
    def __init__(self, sampler):
        self.sampler = sampler

    def exportGeojson(self, outputPath, crs="EPSG:4326"):
        data = []
        nStreets = len(self.sampler.streets)

        for i in range(nStreets):
            street = self.sampler.streets[i]

            # Get the segments
            segmentPoints = street.streetSegments

            for segment in segmentPoints:
                geometry = LineString(segment)  # Treat each segment separately

                # Prepare row data, keeping the same attributes
                rowData = {
                    "name": street.streetId,
                    "length": street.getCompleteLength(),
                    "geometry": geometry
                }

                # Convert any NumPy arrays to lists
                for key, value in street.attributes.items():
                    if isinstance(value, np.ndarray):
                        rowData[key] = value.tolist()  # Convert arrays to lists
                    else:
                        rowData[key] = value  # Keep scalars as is

                data.append(rowData)

        # Create a GeoDataFrame with explicit column definitions
        gdf = gpd.GeoDataFrame(data, geometry="geometry")

        # Set CRS
        gdf = gdf.set_crs(crs)
        print(gdf.dtypes)
        print(gdf.head())

        # Export to GeoJSON
        gdf.to_file(outputPath, driver="GeoJSON")

        print("Successsssssssssssssss, maybe everything isn't hopeless bullshit")