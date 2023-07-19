import GOES
from constants import PATH_OUT, START_DATE, END_DATE

GOES.download('goes16', 'ABI-L2-CMIPF',
                      DateTimeIni = START_DATE, DateTimeFin = END_DATE, 
                      channel = ['01-03', '05', '08', '10', '12', '13'], path_out=PATH_OUT)



GOES.download('goes16', 'GLM-L2-LCFA',
                      DateTimeIni = START_DATE, DateTimeFin = END_DATE,
                      path_out=PATH_OUT)

GOES.download('goes16', 'ABI-L1b-RadF',
                      DateTimeIni = START_DATE, DateTimeFin = END_DATE,
                      channel = ['01','08','13'],
                      path_out=PATH_OUT)

GOES.download('goes17', 'ABI-L2-CMIPF',
                      DateTimeIni = START_DATE, DateTimeFin = END_DATE, 
                      channel = ['08', '13'], path_out=PATH_OUT)


GOES.download('goes16', 'ABI-L2-RRQPEF',
                      DateTimeIni = START_DATE, DateTimeFin = END_DATE,
                      path_out=PATH_OUT)


GOES.download('goes16', 'GLM-L2-LCFA',
                      DateTimeIni = START_DATE, DateTimeFin = END_DATE,
                      path_out=PATH_OUT)


