from datetime import datetime
import rasterio as rio
import pandas as pd
import numpy as np

DRIVE = r'G:/'
GW_PATH = r'GruV-Net/data/groundwater_levels_preprocessed_hesse/weekly/gw_lvl_2008-2017.csv'

class TargetHandlerInterface:
    def from_id_and_date(self, station: str, date: datetime.date, offset: int) -> float:
        pass


class XtremeClassificationHandler(TargetHandlerInterface):

    NAME = 'XtremeClass'

    def __init__(self, data_drive=DRIVE, path=GW_PATH, offset: int=1, lower_quantile=0.05, upper_quantile=0.95) -> None:
        gw_lvl = pd.read_csv(data_drive + path, index_col=0)
        gw_lvl.index = pd.to_datetime(gw_lvl.index).date
        gw_change = gw_lvl.diff(periods=offset)
        lower_bound = gw_change.quantile(lower_quantile)
        upper_bound = gw_change.quantile(upper_quantile)
        self.gw_xtreme = gw_change.copy()
        for column in gw_change:
            self.gw_xtreme[column] = self.gw_xtreme[column].mask(self.gw_xtreme[column] < lower_bound[column], -1)
            self.gw_xtreme[column] = self.gw_xtreme[column].mask(self.gw_xtreme[column] > upper_bound[column], 1)
            self.gw_xtreme[column] = self.gw_xtreme[column].mask((self.gw_xtreme[column] != -1)&(self.gw_xtreme[column] != 1)&(self.gw_xtreme[column].notna()), 0)

    def from_id_and_date(self, station: str, date: datetime.date) -> int:
        return self.gw_xtreme.loc[date, station]