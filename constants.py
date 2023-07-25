from datetime import datetime, timedelta
import os

cwd = os.getcwd()


IMAGES_FOLDER = os.path.join(cwd, "plot_images")
PATH_OUT = os.path.join(cwd, "data")


# Get the current date and time
current_date = datetime.utcnow()

START_DATE = current_date.replace(minute=0, second=0, microsecond=0) - timedelta(minutes=60)

# Format the start date in the desired format
START_DATE = START_DATE.strftime("%Y%m%d-%H%M%S")


END_DATE = current_date.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=61)
END_DATE = END_DATE.strftime("%Y%m%d-%H%M%S")
