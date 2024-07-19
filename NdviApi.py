import urllib.error
import pandas as pd
import numpy as np
import datetime as dt
import urllib

class NDVI(object):
 

    def __init__(self):
        pass
    """
    https://glam1.gsfc.nasa.gov/

    Get the database from here: https://glam1.gsfc.nasa.gov/api/doc/db/versions#default-db
    Just copy and paste the ID as the mask parameter

    The function by itself will return a dataframe with 10 different columns, with samples taken every 8 days
    ORDINAL DATE (as index), START DATE, END DATE, SOURCE, SAMPLE VALUE, SAMPLE COUNT, MEAN VALUE, MEAN COUNT, ANOM VALUE, MIN VALUE, MAX VALUE

    If there is a specific one you want, pass that as col, otherwise pass "all"

    All parameters:
        version: This is the version of the database to use. "v15" is the most recent one

        sat: "MOD", "MYD" or "VNP"

        shape: The shape of the vector you want, can either be ADM or LIS. 

        mask: This is the specific crop database you want get data from, which is obtained from the link above

        start_yr, end_yr, start_month: Self explanatory (note that all three are inclusive)

        num_months: How many months of data you want to get (starting from start_month - 12 for the whole year)

        ids: Pertains to the specificity of the data you want (what admin level). You can find the ids for various admin levels under "Shape IDs" using the link above. 
        Just copy and paste the ID

        ts_type: Time series type for NDVI data: cumulative, complete, or seasonal. 

        mcv: Minimum cumulative value, which is required if ts_type=cumulative. This is the minimum NDVI value to accumulate (valid range is 0.0 - 1.0)

    """
    def get_data(self, 
                 version:str, 
                 sat: str,
                 mask:str, 
                 shape:str,
                 start_yr:int,
                 end_yr:int,
                 start_month:int,
                 num_months:int,
                 ids:int,
                 ts_type:str,
                 mcv:int
                 ):
            try:
                if (ts_type == "cumulative"):
                    df = pd.read_csv(f"https://glam1.gsfc.nasa.gov/api/gettbl/v4?sat={sat}&version={version}&layer=NDVI&mask={mask}&shape={shape}&ids={ids}&ts_type={ts_type}&mcv={mcv}&start_month={start_month}&num_months={num_months}&format=csv", skiprows=14).ffill()
                else:
                    df = pd.read_csv(f"https://glam1.gsfc.nasa.gov/api/gettbl/v4?sat={sat}&version={version}&layer=NDVI&mask={mask}&shape={shape}&ids={ids}&ts_type={ts_type}&start_month={start_month}&num_months={num_months}&format=csv", skiprows=14).ffill()
                df = df[df["ORDINAL DATE"].ge(f"{start_yr}-01-01") & df["ORDINAL DATE"].lt(f"{end_yr + 1}-01-01")]
                return df 
            except urllib.error.HTTPError as err:
                 if err.code == 400:
                      print("The data you requested doesn't exist, meaning there is something wrong with your input parameters")
                 elif err.code == 500:
                      print("If you get this error the site evidently blew up so you will have to wait to get data")

# Seasonal ndvi corn data for 2000-2023 (inclusive) for the whole year
corn_ndvi_data = NDVI().get_data(version="v15", sat="MOD", shape="ADM",
                                 mask="USDA-NASS-CDL_2018-2023_corn-50pp", 
                                 start_yr=2000, end_yr=2023, 
                                 start_month=1, num_months=12, 
                                 ids=27258, ts_type='seasonal', 
                                 mcv=None)
