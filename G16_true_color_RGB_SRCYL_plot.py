# %% [markdown]
# # True Color RGB (RGB example using reprojected VIS channels)
# This jupyter notebook shows how to make a sub-region plot of True color RGB composition with reprojected data. This is an example of RGB composition using reprojected VIS channels of GOES-16.

# %% [markdown]
# **Warning: if your RAM is less than 8GB it is recommended to work with a small domain.**

# %% [markdown]
# Import the GOES package.

# %%
import GOES

# %% [markdown]
# Set path and name of file that will be read.

# %%
from constants import IMAGES_FOLDER, PATH_OUT, START_DATE, END_DATE

C01_files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L2-CMIPF-M*C01_G16*.nc', START_DATE, END_DATE)
C02_files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L2-CMIPF-M*C02_G16*.nc', START_DATE, END_DATE)
C03_files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L2-CMIPF-M*C03_G16*.nc', START_DATE, END_DATE)

C01_file = C01_files[0]
C02_file = C02_files[0]
C03_file = C03_files[0]

# %% [markdown]
# Reads the file.

# %%
C01_ds = GOES.open_dataset(C01_file)
C02_ds = GOES.open_dataset(C02_file)
C03_ds = GOES.open_dataset(C03_file)

# %% [markdown]
# Set the map domain.

# %%
domain = [-90.0,-30.0,-60.0,15.0]

# %% [markdown]
# Creates a grid map with cylindrical equidistant projection and 1.0 km of spatial resolution.

# %%
LonCenCyl, LatCenCyl = GOES.create_gridmap(domain, PixResol=1.0)

# %% [markdown]
# Calculates the coordinates of corners of pixels.

# %%
LonCorCyl, LatCorCyl = GOES.calculate_corners(LonCenCyl, LatCenCyl)

# %% [markdown]
# Calculates the parameters for reprojection. For this we need install the **pyproj** and **pyresample** packages. Try with ***pip install pyproj*** and ***pip install pyresample***.

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
# Gets C01 image with the coordinates of center of their pixels.

# %%
C01, LonCen, LatCen = C01_ds.image('CMI', lonlat='center', domain=domain)

# %% [markdown]
# Reprojects C01 image.

# %%
import numpy as np
from pyresample.geometry import SwathDefinition
from pyresample.kd_tree import resample_nearest

SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
C01Cyl = resample_nearest(SwathDef, C01.data, AreaDef, radius_of_influence=6000,
                          fill_value=np.nan, epsilon=3, reduce_data=True)

# %% [markdown]
# Deletes unnecessary data.

# %%
del C01, LonCen, LatCen

# %% [markdown]
# Gets C02 image with the coordinates of center of their pixels.

# %%
C02, LonCen, LatCen = C02_ds.image('CMI', lonlat='center', domain=domain)

# %% [markdown]
# Reprojects C02 image.

# %%
SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
C02Cyl = resample_nearest(SwathDef, C02.data, AreaDef, radius_of_influence=6000,
                          fill_value=np.nan, epsilon=3, reduce_data=True)

# %% [markdown]
# Deletes unnecessary data.

# %%
del C02, LonCen, LatCen

# %% [markdown]
# Gets C03 image with the coordinates of center of their pixels.

# %%
C03, LonCen, LatCen = C03_ds.image('CMI', lonlat='center', domain=domain)

# %% [markdown]
# Reprojects C03 image.

# %%
SwathDef = SwathDefinition(lons=LonCen.data, lats=LatCen.data)
C03Cyl = resample_nearest(SwathDef, C03.data, AreaDef, radius_of_influence=6000,
                          fill_value=np.nan, epsilon=3, reduce_data=True)

# %% [markdown]
# Gets information about data.

# %%
sat = C01_ds.attribute('platform_ID')
time = C01_ds.variable('time_bounds').data[0]

# %% [markdown]
# Deletes unnecessary data.

# %%
del C03, LonCen, LatCen, SwathDef

# %% [markdown]
# Defines the name of product.

# %%
product = 'True color RGB'

# %% [markdown]
# Calculates the cosine of zenith angle.

# %%
cosz = GOES.cosine_of_solar_zenith_angle(LonCenCyl.data, LatCenCyl.data, time)

# %% [markdown]
# Deletes unnecessary data.

# %%
del LonCenCyl, LatCenCyl

# %% [markdown]
# Makes the RGB composition.

# %%
# converts reflectance factor of channel 01, 02 and 03 to reflectance and save them as R, G and B
R = C02Cyl/cosz.data
G = C03Cyl/cosz.data
B = C01Cyl/cosz.data

# sets limits between 0 and 1
R = np.clip(R, 0, 1)
G = np.clip(G, 0, 1)
B = np.clip(B, 0, 1)

# perfoms a gamma correction
gamma = 2.2
R = np.power(R, 1/gamma)
G = np.power(G, 1/gamma)
B = np.power(B, 1/gamma)

# calculates the true green
G = 0.45*R + 0.1*G + 0.45*B
G = np.clip(G, 0, 1)

# stack channels
RGB = np.dstack((R, G, B)).astype(np.float16)

# reshape RGB to 1-D array
RGB = RGB.reshape((RGB.shape[0]*RGB.shape[1],RGB.shape[2]))

# ensures RGB data is between 0 and 1
RGB = np.clip(RGB, 0.0, 1.0, dtype=RGB.dtype)

# creates a simple array to make the plot with pcolormesh
mask = np.where(np.isnan(C01Cyl)==True, np.nan, 1).astype(np.float16)

# %% [markdown]
# Deletes unnecessary data.

# %%
del C01Cyl, C02Cyl, C03Cyl, cosz, R, G, B

# %% [markdown]
# Creates plot.

# %%
# import packages
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

# calculates the central longitude of the plot
lon_cen = 360.0+(domain[0]+domain[1])/2.0

# creates the figure
fig = plt.figure('map', figsize=(4,4), dpi=200)
ax = fig.add_axes([0.1, 0.16, 0.80, 0.75], projection=ccrs.PlateCarree(lon_cen))
# ax.outline_patch.set_linewidth(0.3)

# add the geographic boundaries
l = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none')
ax.add_feature(l, edgecolor='gold', linewidth=0.25)

# plot the data
img = ax.pcolormesh(LonCorCyl.data, LatCorCyl.data, mask, transform=ccrs.PlateCarree(), color=RGB)

# set the title
ax.set_title('{} - {}'.format(sat, product), fontsize=7, loc='left')
ax.set_title(time.strftime('%Y/%m/%d %H:%M UTC'), fontsize=7, loc='right')

# Sets X axis characteristics
dx = 15
xticks = np.arange(domain[0], domain[1]+dx, dx)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(dateline_direction_label=True))
ax.set_xlabel('Longitude', color='black', fontsize=7, labelpad=3.0)

# Sets Y axis characteristics
dy = 15
yticks = np.arange(domain[2], domain[3]+dy, dy)
ax.set_yticks(yticks, crs=ccrs.PlateCarree())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_ylabel('Latitude', color='black', fontsize=7, labelpad=3.0)

# Sets tick characteristics
ax.tick_params(left=True, right=True, bottom=True, top=True,
               labelleft=True, labelright=False, labelbottom=True, labeltop=False,
               length=0.0, width=0.05, labelsize=5.0, labelcolor='black')

# Sets grid characteristics
ax.gridlines(xlocs=xticks, ylocs=yticks, alpha=0.6, color='gray',
             draw_labels=False, linewidth=0.25, linestyle='--')

# set the map limits
ax.set_extent([domain[0]+360.0, domain[1]+360.0, domain[2], domain[3]], crs=ccrs.PlateCarree())


timestamp = time.strftime('%Y/%m/%d %H:%M UTC')

# Save the plot as an image file with the timestamp as the name
filename = f"G16_true_color_RGB_SRCYL_plot_{timestamp}.png"  # Example filename: plot_20230714123456.png
plt.savefig(f"{IMAGES_FOLDER}/{filename}")


