# %% [markdown]
# # Convert L1b data to L2 data of a VIS channel and make its plot
# This jupyter notebook shows how to convert L1b data to L2 data of a VIS channel of GOES-16 and make a sub-region plot.

# %% [markdown]
# Set path and name of file that will be read.
import GOES
# %%
from constants import IMAGES_FOLDER, PATH_OUT, START_DATE, END_DATE
files = GOES.locate_files(PATH_OUT,
                          'OR_ABI-L1b-RadF-M*C01_G16*.nc', START_DATE, END_DATE)
file = files[0]

# %% [markdown]
# Import the GOES package.

# %%

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
# Gets image with the coordinates of corners of their pixels. In this case the image is a **L1b** data (**Rad**) and it is converted to **L2** data (**CMI**).

# %%
CMI, LonCen, LatCen = ds.image('Rad', lonlat='center', domain=domain, up_level=True)

# %% [markdown]
# Converts reflectance factor to reflectance. This conversion consists of dividing the reflectance factor by the cosine of the zenith angle in each pixel of image.

# %%
CMI = CMI.refl_fact_to_refl(LonCen, LatCen)

# %% [markdown]
# Calculates the coordinates of corners of pixels.

# %%
LonCor, LatCor = GOES.calculate_corners(LonCen, LatCen)

# %% [markdown]
# Deletes the coordinates of centers of pixels since they will no longer be used.

# %%
del LonCen, LatCen

# %% [markdown]
# Gets information about data.

# %%
sat = ds.attribute('platform_ID')
band = ds.variable('band_id').data[0]
wl = ds.variable('band_wavelength').data[0]

# %% [markdown]
# Set the cmap and its characteristics. In this case, we will create a custom color palette using the [custom_color_palette](https://github.com/joaohenry23/custom_color_palette) package.

# %%
# import packages
import custom_color_palette as ccp
import matplotlib.pyplot as plt

# set the colors of the custom palette
paleta = [ plt.cm.Greys_r,  ccp.range(0.0,1.0,0.01) ]

# pass parameters to the creates_palette module
cmap, cmticks, norm, bounds = ccp.creates_palette([paleta], extend='both')

# creating colorbar labels
ticks = ccp.range(0.0,1.0,0.1)

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
filename = f"G16_L1b_to_L2_VIS_SR_plot_{timestamp}.png"  # Example filename: plot_20230714123456.png
plt.savefig(f"{IMAGES_FOLDER}/{filename}")


