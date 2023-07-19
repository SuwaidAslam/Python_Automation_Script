# %% [markdown]
# # Plot the flash product of the GLM
# This jupyter notebook shows how to make a sub-region plot of the flash product of the GLM.

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
# Gets time interval between first and last file.

# %%
time_bounds = ds.variable('product_time_bounds')
time_start = time_bounds.data[0,0]
time_end = time_bounds.data[-1,-1]

# %% [markdown]
# Gets time of first and last event of flash product.

# %%
ti = ds.variable('flash_time_offset_of_first_event')
tf = ds.variable('flash_time_offset_of_last_event')
flash_time = ti.data+(tf.data-ti.data)/2.0

# %% [markdown]
# Converts date and time to numerical format.

# %%
from matplotlib.dates import date2num
flash_time_num = date2num(flash_time)

# %% [markdown]
# Gets information about data.

# %%
sat = ds.attribute('platform_ID')[0]

# %% [markdown]
# Sets product name.

# %%
name = 'Flash occurrence'

# %% [markdown]
# Creates a custom color palette using the [custom_color_palette](https://github.com/joaohenry23/custom_color_palette) package.

# %%
# import packages
import custom_color_palette as ccp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ncols = 6

# set the colors of the custom palette
palette = [['gold','orange','red','darkviolet','violet'], ccp.range(1,ncols,1)]

# pass parameters to the creates_palette module
cmap, cmticks, norm, bounds = ccp.creates_palette([palette], extend='both')

# creates ticks for colorbar
ticks = np.linspace(date2num(time_start),date2num(time_end),ncols)

# calculates norm
norm = Normalize(vmin=ticks[0], vmax=ticks[-1])

# %% [markdown]
# Creates plot.

# %%
# import packages
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.dates import DateFormatter

# calculates the central longitude of the plot
lon_cen = 360.0+(domain[0]+domain[1])/2.0

# creates the figure
fig = plt.figure('map', figsize=(4,4), dpi=200)
ax = fig.add_axes([0.1, 0.16, 0.80, 0.75], projection=ccrs.PlateCarree(lon_cen))
# ax.outline_patch.set_linewidth(0.3)

# add the geographic boundaries
l = NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', facecolor='none')
ax.add_feature(l, edgecolor='black', linewidth=0.25)

# plot the data
img = ax.scatter(flash_lon.data, flash_lat.data, s=0.15, c=flash_time_num, cmap=cmap, norm=norm,
                 marker='o', transform=ccrs.PlateCarree())

# add the colorbar
cb = plt.colorbar(img, ticks=ticks, format=DateFormatter('%H:%M:%S\n%Y/%m/%d'), extend='both',
                    spacing='proportional', orientation='horizontal',
                    cax=fig.add_axes([0.12, 0.05, 0.76, 0.02]))
cb.ax.tick_params(labelsize=5, labelcolor='black', width=0.5, direction='out', pad=1.0)
cb.outline.set_linewidth(0.5)

# set the title
ax.set_title('{} - {}'.format(sat, name), fontsize=5, loc='left')
ax.set_title('{:%Y/%m/%d %H:%M UTC} - {:%Y/%m/%d %H:%M UTC}'.format(time_start, time_end),
             fontsize=5, loc='right')

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
timestamp = '{:%Y/%m/%d %H:%M UTC} - {:%Y/%m/%d %H:%M UTC}'.format(time_start, time_end)

# Save the plot as an image file with the timestamp as the name
filename = f"G16_GLM_FLASH_SR_plot_{timestamp}.png"  # Example filename: plot_20230714123456.png
plt.savefig(f"{IMAGES_FOLDER}/{filename}")



