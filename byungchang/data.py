import numpy as np
import pandas as pd

input_path = '../../input/서울의대법의학교실_Ancestry SNP Panel Genotype raw data'
regions = ['NEA', 'SEA', 'SWA']
popls = [
    'CHS', 'CHB', 'CDX', 'JPT', 'KOR', 
    'KHV', 'VNK', 'MYA',
    'BEB', 'ITU', 'GIH', 'STU', 'NP', 'IN', 'PJL', 'PT'
]

def load_data_v1():
    df_NEA = pd.read_csv(f'{input_path}/서울의대법의학교실_Ancestry SNP Panel Genotype raw data_2022_NEA.csv').dropna(how='all')
    df_NEA.iloc[545:554, 2:] = df_NEA.iloc[545:554, 1:-1]
    df_NEA.iloc[545:554, 1] = 'NN'

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
    df = df.sort_values(by='SNU-ID', key=lambda ser:ser.map(lambda x: popls.index(x)))

    return df


def load_data(version='latest'):
    if version == 'v1':
        return load_data_v1()
    # SNU-KOR-306 vs SNU-KOR-307 rs8035124에서 전자가 NN인데 전자가 중복인 것 같슴다

    # PJL 062번까지는 서로 뒤섞임, 새로 추가된 96명은 뒷번호
    # NP 092번부터 118번까지 서로 뒤섞임
    df = pd.read_csv(f'{input_path}/서울의대 법의학교실 Ancestry SNP genotype (2023-02-10).csv')
    df.loc[df['POP-ID'] == 'PJ', 'POP-ID'] = 'PJL'
    return df
