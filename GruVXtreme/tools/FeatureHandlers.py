from datetime import datetime
from typing import List
import rasterio as rio
import pandas as pd
import numpy as np

DRIVE = r'G:/'
REGNIE_PATH = r'GruV-Net/data/precipitation/weekly-sum/GTiff/'
DWDAIRTEMP_PATH = r'GruV-Net/data/temperature/GTiff/'
STAMMDATEN_PATH = r'GruV-Net/data/Stammdaten_HE.xlsx'
GW_PATH = r'GruV-Net/data/groundwater_levels_preprocessed_hesse/weekly/gw_lvl_2008-2017.csv'


class DynamicFeatureHandlerInterface:
    def from_id_and_date(self, stations: List[str], date: datetime.date) -> List:
        pass


class StaticFeatureHandlerInterface:
    def from_id(self, station: str):
        pass


class DummyFeatureHandler():

    NAME = 'Dummy'

    def __init__(self, val) -> None:
        self.val = val
        self.NAME = f'Dummy{val}-'
    
    def from_id(self, station: str):
        return self.val

    def from_id_and_date(self, station: str, date: datetime.date, raster_size: int):
        return self.val


class UTMEastingHandler(StaticFeatureHandlerInterface):

    NAME = 'UTMEasting'

    def __init__(self, data_drive=DRIVE, path=STAMMDATEN_PATH) -> None:
        self.sd = pd.read_excel(data_drive + path)
        self.sd.set_index('Proj_ID', inplace=True)

    def from_id(self, station: str) -> float:
        easting = self.sd.loc[station]['OSTWERT']
        return easting


class UTMNorthingHandler(StaticFeatureHandlerInterface):

    NAME = 'UTMNorthing'

    def __init__(self, data_drive=DRIVE, path=STAMMDATEN_PATH) -> None:
        self.sd = pd.read_excel(data_drive + path)
        self.sd.set_index('Proj_ID', inplace=True)

    def from_id(self, station: str) -> float:
        northing = self.sd.loc[station]['NORDWERT']
        return northing


class XtremeBoundHandler(StaticFeatureHandlerInterface):

    NAME = 'XtremeBound'

    def __init__(self, data_drive=DRIVE, path=GW_PATH, offset: int=1, quantile=0.5) -> None:
        gw_lvl = pd.read_csv(data_drive + path, index_col=0)
        gw_lvl.index = pd.to_datetime(gw_lvl.index).date
        gw_change = gw_lvl.diff(periods=offset)
        self.NAME = self.NAME + str(quantile)
        self.bound = gw_change.quantile(quantile)

    def from_id(self, station: str) -> int:
        return self.bound[station]


class RegnieHandler(DynamicFeatureHandlerInterface):

    NAME = 'Regnie'

    def __init__(self, data_drive=DRIVE, path=REGNIE_PATH, raster_size: int=11, raster_func=None) -> None:
        self.data_drive = data_drive
        self.path = path
        self.raster_size = raster_size
        self.raster_func = raster_func
        self.EastingHandler = UTMEastingHandler()
        self.NorthingHandler = UTMNorthingHandler()

    def from_id_and_date(self, stations: List[str], date: datetime.date) -> List:
        try:
            src = rio.open(self.data_drive + self.path + 'rws_' + date.strftime('%Y-%m-%d') + '.tif')
        except rio.errors.RasterioIOError:
            return [np.nan]
        res = []
        for s in stations:
            easting, northing = self.EastingHandler.from_id(s), self.NorthingHandler.from_id(s)
            row, col = src.index(easting, northing)
            try:
                res.append(src.read(1, window=rio.windows.Window(col-(np.floor(self.raster_size / 2)), row-(np.floor(self.raster_size / 2)), self.raster_size, self.raster_size)).reshape((self.raster_size,self.raster_size)))
            except ValueError:
                res.append(np.nan)
        if self.raster_func != None: return list(map(self.raster_func,res))
        else: return res


class DWDAirTempHandler(DynamicFeatureHandlerInterface):

    NAME = 'DWDAirTemp'

    def __init__(self, data_drive=DRIVE, path=DWDAIRTEMP_PATH, raster_size: int=11, raster_func=None) -> None:
        self.data_drive = data_drive
        self.path = path
        self.raster_size = raster_size
        self.raster_func = raster_func
        self.EastingHandler = UTMEastingHandler()
        self.NorthingHandler = UTMNorthingHandler()

    def from_id_and_date(self, stations: List[str], date: datetime.date) -> List:
        try:
            src = rio.open(self.data_drive + self.path + 'TAMM_' + date.strftime('%m') + '_' + date.strftime('%Y') + '_01.tif')
        except rio.errors.RasterioIOError:
            return [np.nan]
        res = []
        for s in stations:
            easting, northing = self.EastingHandler.from_id(s), self.NorthingHandler.from_id(s)
            row, col = src.index(easting, northing)
            try:
                res.append(src.read(1, window=rio.windows.Window(col-(np.floor(self.raster_size / 2)), row-(np.floor(self.raster_size / 2)), self.raster_size, self.raster_size)).reshape((self.raster_size,self.raster_size)))
            except ValueError:
                res.append(np.nan)
        if self.raster_func != None: return list(map(self.raster_func,res))
        else: return res


class GwLvlChangeHandler(DynamicFeatureHandlerInterface):

    NAME = 'GwLvlChange'

    def __init__(self, data_drive=DRIVE, path=GW_PATH, offset: int=1) -> None:
        gw_lvl = pd.read_csv(data_drive + path, index_col=0)
        gw_lvl.index = pd.to_datetime(gw_lvl.index).date
        self.gw_change = gw_lvl.diff(periods=offset)
        
    def from_id_and_date(self, stations: List[str], date: datetime.date) -> List:
        res = []
        res.extend(self.gw_change.loc[date, s] for s in stations)
        return res