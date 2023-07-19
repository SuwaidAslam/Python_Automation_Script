from datetime import datetime, timedelta
import os

cwd = os.getcwd()


IMAGES_FOLDER = os.path.join(cwd, "plot_images")
PATH_OUT = os.path.join(cwd, "data")
# Get the current date and time
current_date = datetime.now()

current_date = current_date - timedelta(hours=5)

# Subtract 5 days from the current date
START_DATE = current_date - timedelta(minutes=31)

# Format the start date in the desired format
START_DATE = START_DATE.strftime("%Y%m%d-%H%M%S")


# Calculate the end date by adding 2 hours and 11 minutes to the current date
END_DATE = current_date
END_DATE = END_DATE.strftime("%Y%m%d-%H%M%S")

