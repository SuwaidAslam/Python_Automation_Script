# %% [markdown]
# # Day cloud phase RGB (RGB example using non-reprojected VIS and IR channels)
# This jupyter notebook shows how to make a sub-region plot of Day cloud phase RGB composition. This is an example of RGB composition using non-reprojected VIS and IR channels of GOES-16.

# %% [markdown]
# **Warning: if your RAM is less than 8GB it is recommended to work with a small domain.**

# %% [markdown]
# Import the GOES package.

# %%
import GOES
from constants import IMAGES_FOLDER, PATH_OUT, START_DATE, END_DATE

# %% [markdown]
# Set path and name of file that will be read.

# %%

C02_files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L2-CMIPF-M*C02_G16*.nc', START_DATE, END_DATE)
C05_files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L2-CMIPF-M*C05_G16*.nc', START_DATE, END_DATE)
C13_files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L2-CMIPF-M*C13_G16*.nc', START_DATE, END_DATE)

C02_file = C02_files[0]
C05_file = C05_files[0]
C13_file = C13_files[0]

# %% [markdown]
# Reads the file.

# %%
C02_ds = GOES.open_dataset(C02_file)
C05_ds = GOES.open_dataset(C05_file)
C13_ds = GOES.open_dataset(C13_file)

# %% [markdown]
# Set the map domain.

# %%
domain = [-90.0,-30.0,-60.0,15.0]

# %% [markdown]
# Gets image with the coordinates of center of their pixels.

# %%
C13, _, _ = C13_ds.image('CMI', lonlat='center', domain=domain)

# %% [markdown]
# Gets the **pixels_limits** parameter of C13. This parameter will be used in the other channels to obtain an imagen with the same domain than C13.

# %%
domain_in_pixels = C13.pixels_limits

# %% [markdown]
# Because the spatial resolution of channel 05 is 1.0 km and channel 13 is 2.0 km, it is necessary to **double** the **domain_in_pixels** values to obtain an image of **channel 05** with the same coverage area as **channel 13**.

# %%
C05_domain_in_pixels = [domain_in_pixels[0]*2, (domain_in_pixels[1]+1)*2-1,
                        domain_in_pixels[2]*2, (domain_in_pixels[3]+1)*2-1]

# %% [markdown]
# Creates a mask that identifies the nan values of C13. This mask will be used to obtain the image of channel 05.

# %%
import numpy as np
mask = np.where(np.isnan(C13.data)==True, True, False)

# %% [markdown]
# Repeats each element of the mask two times to match with the image size of channel 05.

# %%
C05_mask = np.kron(mask, np.ones((2,2), dtype=mask.dtype))

# %% [markdown]
# Gets the image of channel 05 with the same coverage area as channel 13.

# %%
C05, LonCen, LatCen = C05_ds.image('CMI', lonlat='center',
                                   domain_in_pixels=C05_domain_in_pixels, nan_mask=C05_mask)

# %% [markdown]
# Since the spatial resolution of channel 02 is 0.5 km and channel 13 is 2.0 km, is necessary to **quadruple** the **domain_in_pixels** values to obtain an image of **channel 01** with the same coverage area as **channel 13**.

# %%
C02_domain_in_pixels = [domain_in_pixels[0]*4, (domain_in_pixels[1]+1)*4-1,
                        domain_in_pixels[2]*4, (domain_in_pixels[3]+1)*4-1]

# %% [markdown]
# Repeats each element of the mask four times to match with the image size of channel 02.

# %%
C02_mask = np.kron(mask, np.ones((4,4), dtype=mask.dtype))

# %% [markdown]
# Gets the image of channel 02 with the same coverage area as channel 01.

# %%
C02, _, _ = C02_ds.image('CMI', lonlat='none', domain_in_pixels=C02_domain_in_pixels, nan_mask=C02_mask)

# %% [markdown]
# Deletes unnecessary arrays.

# %%
del domain_in_pixels, mask
del C05_domain_in_pixels
del C02_domain_in_pixels, C02_mask

# %% [markdown]
# The spatial resolution of C02, C05 y C13 is 0.5 Km, 1.0 Km and 2.0 Km respectively, for this reason the size of C05 is twice the size of C13 and the size of C02 is twice the size of C05. To solve this problem we are going to repeat each pixel in C13 twice and reduce the number of pixels in C02 by averaging groups of 4 pixels (2 on the y axis and 2 on the x axis). This way, C02 and C13 will have the same size as C05.

# %%
# repeat pixels of C13
C13.data = np.kron(C13.data, np.ones((2,2), dtype=C13.data.dtype))

# averaging pixels of C02
nx, ny = 2, 2
ysize, xsize = C02.data.shape
C02.data = np.mean(C02.data.reshape((int(ysize/ny),ny,int(xsize/nx),nx)), axis=(1,3))

# %% [markdown]
# Gets information about data.

# %%
sat = C13_ds.attribute('platform_ID')
time = C13_ds.variable('time_bounds').data[0]

# %% [markdown]
# Defines the name of product.

# %%
product = 'Day snow-fog RGB'

# %% [markdown]
# Calculates the cosine of zenith angle.

# %%
cosz = GOES.cosine_of_solar_zenith_angle(LonCen.data, LatCen.data, time)

# %% [markdown]
# Calculates the coordinates of corners of pixels.

# %%
LonCor, LatCor = GOES.calculate_corners(LonCen, LatCen)

# %% [markdown]
# Deletes the coordinates of centers of pixels since they will no longer be used.

# %%
del LonCen, LatCen

# %% [markdown]
# Makes the RGB composition.

# %%
# converts reflectance factor of channel 01, 02 and 03 to reflectance and save them as R, G and B
R = C13.data - 273.15
G = C02.data/cosz.data
B = C05.data/cosz.data

# set limits of channels
Rmin, Rmax = -53.5, 7.5
Gmin, Gmax = 0.0, 0.78
Bmin, Bmax = 0.1, 0.59
R = np.clip(R, Rmin, Rmax)
G = np.clip(G, Gmin, Gmax)
B = np.clip(B, Bmin, Bmax)

# normalize channels
R = (R-Rmin)/(Rmax-Rmin)
G = (G-Gmin)/(Gmax-Gmin)
B = (B-Bmin)/(Bmax-Bmin)

# perfoms a gamma correction
gamma = 1.0
R = np.power(R, 1/gamma)
G = np.power(G, 1/gamma)
B = np.power(B, 1/gamma)

# invert the channel
R = 1.0 - R

# stack channels
RGB = np.dstack((R, G, B)).astype(np.float16)

# reshape RGB to 1-D array
RGB = RGB.reshape((RGB.shape[0]*RGB.shape[1],RGB.shape[2]))

# ensures RGB data is between 0 and 1
RGB = np.clip(RGB, 0.0, 1.0, dtype=RGB.dtype)

# creates a simple array to make the plot with pcolormesh
mask = np.where(C05_mask==True,np.nan, 1).astype(np.float16)

# %% [markdown]
# Deletes unnecessary data.

# %%
del C13, C02, C05, cosz, R, G, B, C05_mask

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
img = ax.pcolormesh(LonCor.data, LatCor.data, mask, transform=ccrs.PlateCarree(), color=RGB)

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
filename = f"G16_day_cloud_phase_RGB_SR_plot_{timestamp}.png"  # Example filename: plot_20230714123456.png
plt.savefig(f"{IMAGES_FOLDER}/{filename}")


