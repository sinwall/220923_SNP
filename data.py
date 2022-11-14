import numpy as np
import pandas as pd

input_path = '../input/서울의대법의학교실_Ancestry SNP Panel Genotype raw data'

def load_data():
    df_NEA = pd.read_csv(f'{input_path}/서울의대법의학교실_Ancestry SNP Panel Genotype raw data_2022_NEA.csv').dropna(how='all')
    df_SEA = pd.read_csv(f'{input_path}/서울의대법의학교실_Ancestry SNP Panel Genotype raw data_2022_SEA.csv').dropna(how='all')
    df_SWA = pd.read_csv(f'{input_path}/서울의대법의학교실_Ancestry SNP Panel Genotype raw data_2022_SWA.csv').dropna(how='all')
    df = pd.concat([df_NEA, df_SEA, df_SWA], axis=0, ignore_index=True).fillna('NN')

    df = pd.concat([
        df, 
        pd.Series(np.concatenate([
            np.full((df_NEA.shape[0], ), 'NEA'), 
            np.full((df_SEA.shape[0], ), 'SEA'),
            np.full((df_SWA.shape[0], ), 'SWA')
        ], axis=0), name='region'),
        df['SNU-ID'].str.split('-').apply(lambda x:x[1]).rename('popl')
    ], axis=1).astype('string')

    return df