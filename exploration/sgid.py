import geopandas as gpd
import matplotlib.pyplot as plt

connection = "postgresql://agrc:agrc@opensgid.ugrc.utah.gov:5432/opensgid"

query = "SELECT name, shape FROM boundaries.county_boundaries"

counties_df= gpd.read_postgis(query, connection, geom_col="shape")

print(counties_df.head())

counties_df.plot(edgecolor="black", facecolor="lightblue", figsize=(8,10))
plt.title("Utah County Boundaries")
plt.show()
