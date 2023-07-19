# %% [markdown]
# # GOES-16 WV channel subregion plot
# This jupyter notebook shows how to make a sub-region plot of a WV channel of GOES-16.

# %% [markdown]
# Set path and name of file that will be read.
# %%
import GOES

# %%
from constants import IMAGES_FOLDER, PATH_OUT, START_DATE, END_DATE
C08_files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L2-CMIPF-M*C08_G16*.nc', START_DATE, END_DATE)
file = C08_files[0]

# %% [markdown]
# Import the GOES package.

# %% [markdown]
# Reads the file.

# %%
ds = GOES.open_dataset(file)

# %% [markdown]
# Prints the contents of the file.

# %%
print(ds)

# %% [markdown]
# Set the map domain.

# %%
domain = [-90.0,-30.0,-60.0,15.0]

# %% [markdown]
# Gets image with the coordinates of corners of their pixels.

# %%
CMI, LonCor, LatCor = ds.image('CMI', lonlat='corner', domain=domain)

# %% [markdown]
# Gets information about data.

# %%
sat = ds.attribute('platform_ID')
band = ds.variable('band_id').data[0]
wl = ds.variable('band_wavelength').data[0]

# %% [markdown]
# Creates a custom color palette using the [custom_color_palette](https://github.com/joaohenry23/custom_color_palette) package.

# %%
# import packages
import custom_color_palette as ccp
import matplotlib.pyplot as plt

# set the colors of the custom palette
pl1 = [[(85/255,0/255,84/255),(174/255,46/255,172/255),(239/255,139/255,238/255)],
       ccp.range(183.0,198,0.5)]
pl2 = [[(0/255,54/255,0/255),'lawngreen'],
       ccp.range(198.0,213.0,0.5)]
pl3 = [['darkblue','white'],
       ccp.range(213.0,228.0,0.5)]
pl4 = [[(240/255,240/255,240/255),(60/255,60/255,60/255)],
       ccp.range(228.0,248.0,0.5), [ccp.range(227.0,248.0,0.5),228.0,248.0]]
pl5 = [[(65/255,36/255,2/255),'orange','red','darkred',(63/255,0/255,0/255),'black'],
       ccp.range(248.0,288.0,0.5)]

# pass parameters to the creates_palette module
cmap, cmticks, norm, bounds = ccp.creates_palette([pl1, pl2, pl3, pl4, pl5], extend='both')

# creating colorbar labels
ticks = ccp.range(183,288,5)

# %% [markdown]
# Creates plot.

# %%
# import packages
import numpy as np
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
img = ax.pcolormesh(LonCor.data, LatCor.data, CMI.data, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

# add the colorbar
cb = plt.colorbar(img, ticks=ticks, orientation='horizontal', extend='both',
                  cax=fig.add_axes([0.12, 0.05, 0.76, 0.02]))
cb.ax.tick_params(labelsize=4, labelcolor='black', width=0.5, length=1.5, direction='out', pad=1.0)
cb.set_label(label='{} [{}]'.format(CMI.standard_name, CMI.units), size=5, color='black', weight='normal')
cb.outline.set_linewidth(0.5)

# set the title
ax.set_title('{} - C{:02d} [{:.1f} μm]'.format(sat,band, wl), fontsize=7, loc='left')
ax.set_title(CMI.time_bounds.data[0].strftime('%Y/%m/%d %H:%M UTC'), fontsize=7, loc='right')

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


timestamp = CMI.time_bounds.data[0].strftime(
    '%Y%m%d%H%M%S UTC')

# Save the plot as an image file with the timestamp as the name
filename = f"G16_WV__SR_plot_{timestamp}.png"  # Example filename: plot_20230714123456.png
plt.savefig(f"{IMAGES_FOLDER}/{filename}")



