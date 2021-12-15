"This page is used to fetch pulse information"

import numpy as np
import pandas as pd


SeedTable = pd.read_excel('Parameters.xls', sheet_name = 0).to_dict(orient='records');
SeedDict = SeedTable[0]

PumpTable = pd.read_excel('Parameters.xls', sheet_name = 1).to_dict(orient='records');
PumpDict = PumpTable[0]








