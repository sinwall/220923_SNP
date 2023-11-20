import numpy as np
import pandas as pd

folder_name = '서울의대법의학교실_Ancestry SNP Panel Genotype raw data'
regions = ['NEA', 'SEA', 'SWA']
popls = [
    'CHS', 'CHB', 'CDX', 'JPT', 'KOR', 
    'KHV', 'VNK', 'MYA',
    'BEB', 'ITU', 'GIH', 'STU', 'NP', 'IN', 'PJL', 'PT'
]
popls = [
    'JPT', 'KOR', 'CHS', 'CHB', 'CDX',
    'KHV', 'VNK', 'MYA',
    'NP', 'BEB', 'GIH', 'IN', 'ITU', 'STU', 'PJL', 'PT'
]
ystr_sizes = [4, 4, 6, 4, 4, 4, 3,  4, 4, 5, 4, 4, 4, 4,  4, 3, 5, 4, 4, 4, 4,  4, 4]
mh_sizes = [3, 2, 4, 4, 4, 3, 3, 4, 3, 4, 2, 5, 3, 4, 3, 2, 4, 3, 5, 4, 4, 2,
       3, 4, 4, 4, 2, 4, 3, 3, 4, 4, 3, 2, 2, 2, 2, 3, 5, 4, 4, 2, 2, 4,
       4, 4, 5, 4, 3, 4, 3, 4, 4, 4, 3, 2]


def load_data_v1(data_dir='../input'):
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


def load_data(data_dir='../input', version='latest'):
    if version == 'v1':
        return load_data_v1(data_dir=data_dir)
    input_path = f'{data_dir}/{folder_name}'
    # SNU-KOR-306 vs SNU-KOR-307 rs8035124에서 전자가 NN인데 전자가 중복인 것 같슴다

    # PJL 062번까지는 서로 뒤섞임, 새로 추가된 96명은 뒷번호
    # NP 092번부터 118번까지 서로 뒤섞임
    df = pd.read_csv(f'{input_path}/서울의대 법의학교실 Ancestry SNP genotype (2023-02-10).csv')
    df.loc[df['POP-ID'] == 'PJ', 'POP-ID'] = 'PJL'
    return df


def load_demography(data_dir='../input'):
    input_path = f'{data_dir}/{folder_name}'
    df = pd.read_csv(f'{input_path}/서울의대 법의학교실 Ancestry SNP genotype (2023-02-10).csv')
    df = df[['Geographic region', 'POP-ID', 'SNU-ID']]
    df.loc[df['POP-ID'] == 'PJ', 'POP-ID'] = 'PJL'
    df = df.set_index('SNU-ID', drop=True)

    return df


def load_data_snp(data_dir='../input'):
    input_path = f'{data_dir}/{folder_name}'
    # SNU-KOR-306 vs SNU-KOR-307 rs8035124에서 전자가 NN인데 전자가 중복인 것 같슴다

    # PJL 062번까지는 서로 뒤섞임, 새로 추가된 96명은 뒷번호
    # NP 092번부터 118번까지 서로 뒤섞임
    df = pd.read_csv(f'{input_path}/서울의대 법의학교실 Ancestry SNP genotype (2023-02-10).csv')
    # df.loc[df['POP-ID'] == 'PJ', 'POP-ID'] = 'PJL'
    df = df.drop(columns=['Geographic region', 'POP-ID'])
    df = df.set_index('SNU-ID', drop=True)

    # manual correction
    df.loc['SNU-VNK-090', 'rs3118378'] = 'NN'
    df.loc[['SNU-GIH-090', 'SNU-GIH-092', 'SNU-GIH-108'], 'rs10954737'] = 'NN'
    df.loc['SNU-CHB-068', 'rs10108270'] = 'NN'
    return df


def load_data_str(data_dir='../input'):
    input_path = f'{data_dir}/{folder_name}'
    df = pd.read_csv(f'{input_path}/★서울의대 법의학교실 Ancestry genotype (2023-10-30) - 수학과_str.csv')

    def parse_entry(x, size):
        if ',' in x:
            return np.nan
            # return np.mean(list(map(int, x.split(','))))
        elif '.' in x:
            a, b = x.split('.')
            return int(a) + int(b)/size
        elif x == 'del':
            return np.nan
        elif x == 'Null':
            return np.nan
        else:
            raise ValueError
    columns = [col for col in df.columns if (col.startswith('DYS') or col.startswith('YGATA'))]
    for col, size in zip(columns, ystr_sizes):
        if not hasattr(df[col], 'str'): continue
        mask_numeric = df[col].str.isnumeric()
        
        df.loc[mask_numeric, col] = df.loc[mask_numeric, col].astype(np.int32)
        df.loc[~mask_numeric, col] = df.loc[~mask_numeric, col].map(lambda x: parse_entry(x, size))
        df[col] = df[col].astype(np.float32)
    
    # df.loc[df['POP-ID'] == 'PJ', 'POP-ID'] = 'PJL'
    df = df.drop(columns=['Geographic region', 'POP-ID'])
    df = df.set_index('SNU-ID', drop=True)
    return df


def load_data_mh(data_dir='../input'):
    input_path = f'{data_dir}/{folder_name}'
    df = pd.read_csv(f'{input_path}/★서울의대 법의학교실 Ancestry genotype (2023-10-30) - 수학과_mh.csv')

    # df.loc[df['POP-ID'] == 'PJ', 'POP-ID'] = 'PJL'
    df = df.drop(columns=['Geographic region', 'POP-ID'])
    df = df.set_index('SNU-ID', drop=True)
    for name in df.columns:
        if (df[name] == 'Null').any():
            size = len(df.loc[df[name] != 'Null', name].iloc[0])
            df.loc[df[name] == 'Null', name] = 'N'*size
    return df