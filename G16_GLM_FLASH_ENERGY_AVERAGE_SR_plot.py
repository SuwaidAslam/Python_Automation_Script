# %% [markdown]
# # Plots the average energy of flashes of GLM
# This jupyter notebook shows how to make a sub-region plot of the average energy of flashes of GLM.

# %% [markdown]
# Import the GOES package.

# %%
import GOES
from constants import IMAGES_FOLDER, PATH_OUT, START_DATE, END_DATE
# %% [markdown]
# Searchs GLM files between **2021-03-13 22:35:00** and **2021-03-13 22:40:00**.

# %%
flist=GOES.locate_files(PATH_OUT, 'OR_GLM*.nc',
                       START_DATE, END_DATE)

# %% [markdown]
# Reads the files.

# %%
ds = GOES.open_mfdataset(flist)

# %% [markdown]
# Prints the contents of the files.

# %%
print(ds)

# %% [markdown]
# Set the map domain.

# %%
domain = [-86.0,-31.0,-40.0,15.0]

# %% [markdown]
# Gets longitude and latitude of flash product of GLM.

# %%
flash_lon = ds.variable('flash_lon')
flash_lat = ds.variable('flash_lat')

# %% [markdown]
# Gets energy of flash product of GLM.

# %%
flash_energy = ds.variable('flash_energy')

# %% [markdown]
# Since the flash energy has units of femtojoules (fJ; **1e-15** J), we are going to multiply it by **1e15** to facilitate its accumulation.

# %%
flash_energy.data = flash_energy.data*1e15

# %% [markdown]
# Gets time interval between first and last file.

# %%
time_bounds = ds.variable('product_time_bounds')
time_start = time_bounds.data[0,0]
time_end = time_bounds.data[-1,-1]

# %% [markdown]
# Creates a regular grid map with **2 km** of spatial resolution. In this example, we are using this spatial resolution, but you can change it according to your needs.

# %%
pix_resol = 2.0
gridmap_LonCor, gridmap_LatCor = GOES.create_gridmap(domain, PixResol=pix_resol)

# %% [markdown]
# Accumulate flash in the gridmap. Keep in mind that **gridmap_LonCor** and **gridmap_LatCor** are the corners of the pixels where the lightnings will accumulate. <br>
# **Notice:** If your version of GOES package not have the **GOES.accumulate_in_gridmap** function or if you have problems to run the following code, then you must update the **GOES** package to the latest version.

# %%
dens = GOES.accumulate_in_gridmap(gridmap_LonCor, gridmap_LatCor, flash_lon, flash_lat)

# %% [markdown]
# Accumulate flash energy in the gridmap.

# %%
energy_accum = GOES.accumulate_in_gridmap(gridmap_LonCor, gridmap_LatCor,
                                          flash_lon, flash_lat, parameter_value=flash_energy)

# %% [markdown]
# Calculates the average flash energy.

# %%
import numpy as np

avg_energy = np.where(dens.data>0,energy_accum.data/dens.data,0.0)

# %% [markdown]
# Gets information about data.

# %%
sat = ds.attribute('platform_ID')[0]

# %% [markdown]
# Sets product name and its unit.

# %%
name = 'Average flash energy'
unit = '1e-15 J'

# %% [markdown]
# Creates a custom color palette using the [custom_color_palette](https://github.com/joaohenry23/custom_color_palette) package.

# %%
# import packages
import custom_color_palette as ccp
import matplotlib.pyplot as plt

# set the colors of the custom palette
pl1 = [['black'], [0,1]]
pl2 = [plt.cm.viridis, [1,100,200,300,400,500,750,1000,1500,2000], [[1,2,3,4,5,6,7,8,9,10,11,12,13],4,13]]

# pass parameters to the creates_palette module
cmap, cmticks, norm, bounds = ccp.creates_palette([pl1,pl2], extend='max')

# set ticks for colorbar
ticks = cmticks

# %% [markdown]
# Creates plot.

# %%
# import packages
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# calculates the central longitude of the plot
lon_cen = 360.0+(domain[0]+domain[1])/2.0

# creates the figure
fig = plt.figure('map', figsize=(4,4), dpi=200)
ax = fig.add_axes([0.1, 0.16, 0.80, 0.75], projection=ccrs.PlateCarree(lon_cen))
# ax.outline_patch.set_linewidth(0.3)

# add the geographic boundaries
l = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none')
ax.add_feature(l, edgecolor='white', linewidth=0.25)

# plot the data
img = ax.pcolormesh(gridmap_LonCor.data, gridmap_LatCor.data, avg_energy, cmap=cmap, norm=norm,
                    transform=ccrs.PlateCarree())

# add the colorbar
cb = plt.colorbar(img, ticks=ticks, extend='max', orientation='horizontal',
                  cax=fig.add_axes([0.12, 0.05, 0.76, 0.02]))
cb.ax.tick_params(labelsize=5, labelcolor='black', width=0.5, direction='out', pad=1.0)
cb.set_label(label='{} [{}]'.format(name, unit), size=5, color='black', weight='normal')
cb.outline.set_linewidth(0.5)

# set the title
ax.set_title('{} - {} [{}x{}km]'.format(sat, name, pix_resol, pix_resol), fontsize=4.5, loc='left')
ax.set_title('{:%Y/%m/%d %H:%M UTC} - {:%Y/%m/%d %H:%M UTC}'.format(time_start, time_end),
             fontsize=4.5, loc='right')

# Sets X axis characteristics
dx = 10
xticks = np.arange(domain[0], domain[1]+dx, dx)
ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(dateline_direction_label=True))
ax.set_xlabel('Longitude', color='black', fontsize=7, labelpad=3.0)

# Sets Y axis characteristics
dy = 10
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

from datetime import datetime
# Get the current timestamp
timestamp = '{:%Y/%m/%d %H:%M UTC} - {:%Y/%m/%d %H:%M UTC}'.format(time_start, time_end)

# Save the plot as an image file with the timestamp as the name
filename = f"G16_GLM_FLASH_ENERGY_AVERAGE_SR_plot_{timestamp}.png"  # Example filename: plot_20230714123456.png
plt.savefig(f"{IMAGES_FOLDER}/{filename}")


