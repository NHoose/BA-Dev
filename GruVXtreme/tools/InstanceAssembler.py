from datetime import datetime, timedelta
from typing import List
import pandas as pd
import numpy as np

"""Stellt Datensätze aus Features und Tergetvalues im tidy data Format zusammen.

stations: Die Stations IDs für die Instanzen erstellt werden sollen
dates: Die Dates für die eine Instanz erstellt werden soll
n_weeks: Die Anzahl an Wochen im Feature-Vektor vor dem Date
stepsize: max. zeitliche Auflösung des Inputs in Wochen
pred_offset: Die Anzahl an Wochen die in die Zukunft vorhergesagt wird, wichtig für die richtige Lände des Feature-Vectors. Bsp: offset = 4 -> Date ist 4 Wochen in Zukunft, Features für Wochen (Date - (n_weeks + offset)) bis (Date - offset)
target_handler: Liefert y-Wert zurück
dynamic_feature_handlers: Handler für alle zu verwendenden dynamischen Features
static_feature_handlers: Handler für alle zu verwendenden statischen Features
"""
def point_data_assembler(stations, dates: List[datetime.date], n_weeks, stepsize, pred_offset, target_handler, dynamic_feature_handlers, static_feature_handlers) -> pd.DataFrame:
    labels = ['Date', 'Station', target_handler.NAME]
    for sfh in static_feature_handlers:
        labels = labels + [sfh.NAME]
    for i in range(int(n_weeks/stepsize)):
        for dfh in dynamic_feature_handlers:
            labels = labels + [dfh.NAME + str(i)]
    
    x = []
    for targetdate in dates:
        static_features = []
        #TODO static Handler und Target Handler auf Liste von Stationen umstellen
        for s in stations:
            y = target_handler.from_id_and_date(s, targetdate)
            static = [targetdate, s, y]
            for sfh in static_feature_handlers:
                f = sfh.from_id(s)
                static.extend([f])
            static_features.append(static)
        dynamic_features = []        
        for w in range(n_weeks+(pred_offset-1), pred_offset-1, -stepsize):
            d = targetdate - timedelta(weeks=w)
            for dfh in dynamic_feature_handlers:
                f = dfh.from_id_and_date(stations, d)
                dynamic_features.append(f)
        x.extend(np.concatenate((static_features, np.transpose(dynamic_features)), axis=1))

    return pd.DataFrame(x, columns=labels)