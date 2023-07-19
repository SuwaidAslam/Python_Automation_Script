# %% [markdown]
# # Save GOES-16 IR channel subregion as Geotiff file
# This jupyter notebook shows how to save a subregion of channel 13 (GOES-16) to GeoTiff file using the GOES, pyresample and GDAL packages. This methodology can be used with other ABI channels and ABI-derived products of GOES-16 and GOES-17. This tutorial consists of the following sections:
# 
# 1- Get data from netcdf <br>
# 2- Resample data <br>
# 3- Save data as geotiff file<br>
# 4- Check the geotiff file<br>

# %% [markdown]
# <a id='get'></a>
# ## 1- Get data from netcdf

# %% [markdown]
# Import the GOES package.

# %%
import GOES

# %% [markdown]
# Set path and name of file that will be read.

# %%
from constants import IMAGES_FOLDER, PATH_OUT, START_DATE, END_DATE
C13_files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L2-CMIPF-M*C13_G16*.nc', START_DATE, END_DATE)
file = C13_files[0]

# %% [markdown]
# Reads the file.

# %%
ds = GOES.open_dataset(file)

# %% [markdown]
# Set the map domain.

# %%
domain = [-85.0,-65.0,-19.0,1.0] #LonMin, LonMax, LatMin, LatMax

# %% [markdown]
# Gets image with the coordinates of center of their pixels.

# %%
CMI, LonCen, LatCen = ds.image('CMI', lonlat='center', domain=domain)

# %% [markdown]
# Gets information about data.

# %%
sat = ds.attribute('platform_ID')
band = ds.variable('band_id').data[0]
wl = ds.variable('band_wavelength').data[0]
standard_name = CMI.standard_name
units = CMI.units
time_bounds = CMI.time_bounds

# %% [markdown]
# <a id='resample'></a>
# ## 2- Resample data
# Since the netcdf data has satellite projection, it is necessary to reproject it to equirectangular projection to can save it as a geotiff. The follow steps teach how to do that.

# %% [markdown]
# Creates a grid map with cylindrical equidistant projection and 2 km of spatial resolution.

# %%
LonCenCyl, LatCenCyl = GOES.create_gridmap(domain, PixResol=2.0)

# %% [markdown]
# Calculates the parameters for reprojection. For this we need install the **pyproj** and **pyresample** packages. Do it writing ***pip install pyproj*** and ***pip install pyresample***.

# %%
import pyproj as pyproj
Prj = pyproj.Proj('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +b=6378.137 +units=km')
AreaID = 'cyl'
AreaName = 'cyl'
ProjID = 'cyl'
Proj4Args = '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +b=6378.137 +units=km'

ny, nx = LonCenCyl.data.shape
SW = Prj(LonCenCyl.data.min(), LatCenCyl.data.min())
NE = Prj(LonCenCyl.data.max(), LatCenCyl.data.max())
area_extent = [SW[0], SW[1], NE[0], NE[1]]

from pyresample import utils
AreaDef = utils.get_area_def(AreaID, AreaName, ProjID, Proj4Args, nx, ny, area_extent)

# %% [markdown]
# Reprojects image.

# %%
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import resample_nearest
import numpy as np

SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
CMICyl = resample_nearest(SwathDef, CMI.data, AreaDef, radius_of_influence=6000,
                          fill_value=np.nan, epsilon=3, reduce_data=True)

# %% [markdown]
# Deletes unnecessary data.

# %%
del CMI, LonCen, LatCen, SwathDef

# %% [markdown]
# <a id='save'></a>
# ## 3- Save data as geotiff file

# %% [markdown]
# To save the data as geotiff we going to creates a functions called **save_as_geotiff**. For this we need install the **gdal** packages. You can install the package with ***conda install -c conda-forge gdal***.

# %%
import gdal, osr

def save_as_geotiff(Field, LonsCen, LatsCen, OutputFileName):
    deltaLon = LonsCen[0,1]-LonsCen[0,0]
    deltaLat = LatsCen[1,0]-LatsCen[0,0]
    LonCor = LonsCen[0,0] - (deltaLon)/2.0
    LatCor = LatsCen[0,0] - (deltaLat)/2.0
    #
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(OutputFileName, Field.shape[1], Field.shape[0], 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((LonCor, deltaLon, 0, LatCor, 0, deltaLat))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(Field)
    #
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

# %% [markdown]
# Saves data as geotiff in **/home/joao/Downloads/** with the name **data.tif**.

# %%
save_as_geotiff(CMICyl, LonCenCyl.data, LatCenCyl.data, '/home/joao/Downloads/data.tif')

# %% [markdown]
# <a id='check'></a>
# ## 4- Check the geotiff file
# Reads the geotiff.

# %%
gtif = gdal.Open('/home/joao/Downloads/data.tif')
geoinformation = gtif.GetGeoTransform()

nx = gtif.RasterXSize
ny = gtif.RasterYSize
xres = geoinformation[1]
yres = geoinformation[5]
ULLon = geoinformation[0]
ULLat = geoinformation[3]
bnd = gtif.GetRasterBand(1).ReadAsArray()

print('Upper left longitude: {}\nUpper left latitude: {}'.format(ULLon, ULLat))
print('Spatial resolution in X (Lon): {}\nSpatial resolution in Y (Lat): {}'.format(xres, yres))
print('Number of pixels in X: {}\nNumber of pixels in Y: {}'.format(nx, ny))

# %% [markdown]
# Calculates latitude and longitude of pixels corners.

# %%
LonsCor = np.arange(bnd.shape[1]+1)*xres+ULLon
LatsCor = np.arange(bnd.shape[0]+1)*yres+ULLat
LonsCor, LatsCor = np.meshgrid(LonsCor, LatsCor)

# %% [markdown]
# Creates the plot.

# %%
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

# calculates the central longitude of the plot
LonCenPlot = LonsCor[0,:].mean() + 360.0

# calculates the extent of the plot
ExtentPlot = [LonsCor[0,0]+360.0, LonsCor[0,-1]+360.0, LatsCor[-1,0], LatsCor[0,0]]

# creates the figure
fig = plt.figure('map', figsize=(4,4), dpi=200)
ax = fig.add_axes([0.1, 0.16, 0.80, 0.75], projection=ccrs.PlateCarree(LonCenPlot))
# ax.outline_patch.set_linewidth(0.3)

# add the geographic boundaries
l = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none')
ax.add_feature(l, edgecolor='gold', linewidth=0.4)

# plot the data
img = ax.pcolormesh(LonsCor, LatsCor, bnd, cmap='gray_r', transform=ccrs.PlateCarree())

# add the colorbar
cb = plt.colorbar(img)

# set the map limits
ax.set_extent(ExtentPlot, crs=ccrs.PlateCarree())

timestamp = CMI.time_bounds.data[0].strftime(
    '%Y%m%d%H%M%S UTC')

# Save the plot as an image file with the timestamp as the name
filename = f"G16_IR__SR_to_Geotiff_{timestamp}.png"  # Example filename: plot_20230714123456.png
plt.savefig(f"{IMAGES_FOLDER}/{filename}")

# %%



