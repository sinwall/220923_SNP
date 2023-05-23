import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans Mono'
from gtda.homology import VietorisRipsPersistence

from sklearn.decomposition import PCA
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline

from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score


import itertools

import csv
import argparse

from data import load_data

class GaussianMixturePP(GaussianMixture):
    def fit(self, X, y):
        super().fit(X, y)
        labels = np.unique(y)
        y_recon = super().predict(X)
        perms = list(
            labels[list(perm)] for perm in itertools.permutations(range(self.n_components))
        )
        scores = np.empty((len(perms), ))
        for i, perm in enumerate(perms):
            y_perm = np.vectorize(perm.__getitem__)(y_recon)
            scores[i] = np.mean(y_perm == y)
        self.optimal_perm = perms[np.argmax(scores)]
        return self
    
    def predict(self, X, y=None):
        return np.vectorize(self.optimal_perm.__getitem__)(super().predict(X))
    

def define_argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument('--filename', default='exp_log', type=str)
    ap.add_argument('--n_iter', default=500, type=int)
    ap.add_argument('--drop_min', default=5, type=int)
    ap.add_argument('--drop_max', default=50, type=int)
    ap.add_argument('--drop_step', default=5, type=int)
    ap.add_argument('--barcode_size', default=10, type=int)

    config = ap.parse_args()

    return config

def _extract_persistence_from_data(arr, metric='hamming', top_k=1):
    pers_hom = VietorisRipsPersistence(
        metric=metric, 
        homology_dimensions=(1, )
    )
    pers = pers_hom.fit_transform(arr[np.newaxis])[0]

    result = np.zeros((top_k, 2))
    result[:pers.shape[0]] = pers[(np.argsort(pers[:, 0] - pers[:, 1]))[:top_k], :2]
    return result

def main(config):
    df = load_data()
    fd = df.transpose().iloc[3:]

    base_freq = np.zeros((df.shape[1]-3, 4))
    for num_order, (name, col) in enumerate(df.iloc[:, 3:].items()):
        base_freq[num_order, 0] = col.str.count('A').sum()
        base_freq[num_order, 1] = col.str.count('C').sum()
        base_freq[num_order, 2] = col.str.count('G').sum()
        base_freq[num_order, 3] = col.str.count('T').sum()
    base_freq_rel = base_freq / np.sum(base_freq, axis=1, keepdims=True)

    mask_AC = np.all(np.argsort(base_freq, axis=1)[:, 2:] == [0, 1], axis=1)
    mask_CA = np.all(np.argsort(base_freq, axis=1)[:, 2:] == [1, 0], axis=1)
    mask_AG = np.all(np.argsort(base_freq, axis=1)[:, 2:] == [0, 2], axis=1)
    mask_GA = np.all(np.argsort(base_freq, axis=1)[:, 2:] == [2, 0], axis=1)
    mask_CT = np.all(np.argsort(base_freq, axis=1)[:, 2:] == [1, 3], axis=1)
    mask_TC = np.all(np.argsort(base_freq, axis=1)[:, 2:] == [3, 1], axis=1)
    mask_GT = np.all(np.argsort(base_freq, axis=1)[:, 2:] == [2, 3], axis=1)
    mask_TG = np.all(np.argsort(base_freq, axis=1)[:, 2:] == [3, 2], axis=1)

    mask_AC_CA = mask_AC | mask_CA
    mask_AG_GA = mask_AG | mask_GA
    mask_CT_TC = mask_CT | mask_TC
    mask_GT_TG = mask_GT | mask_TG

    X_str = df.iloc[:, 3:].values
    X_oe = OrdinalEncoder().fit_transform(X_str)
    y = df['Geographic region'].values
    min_categories = df.iloc[:, 3:].nunique().values


    n_iter = config.n_iter
    top_k = config.barcode_size


    dist_f = np.zeros((fd.shape[0], fd.shape[0]))
    for i, j in np.ndindex(dist_f.shape):
        dist_f[i, j] = (fd.iloc[i] != fd.iloc[j]).mean()
        dist_f[j, i] = dist_f[i, j]

    base3_with_perm = np.zeros((24, fd.shape[0], fd.shape[1]))
    for num_order, order in enumerate(itertools.permutations('ACGT')):
        for num_row, (_, row) in enumerate(fd.iterrows()):
            for base in 'ACGT':
                base3_with_perm[num_order, num_row] += row.str.count(base) * (3**order.index(base))

    dist_q = np.zeros((fd.shape[0], fd.shape[0]))
    for i, j in np.ndindex(dist_q.shape):
        if i >= j: continue
        dist_q[i, j] = np.min(np.mean(base3_with_perm[0, i] != base3_with_perm[:, j], axis=-1))
        dist_q[j, i] = dist_q[i, j]

    file_w = open(f'../output/20230517_{config.filename}.csv', 'w', newline='')
    writer = csv.writer(file_w)
    colnames = ['A&G_only', 'n_drop', 'split_id', 'model'] + \
        ['accuracy_control', 'accuracy_modified']
    for i in range(top_k):
        colnames += [f'birth_control_{i}', f'death_control_{i}']
    for i in range(top_k):
        colnames += [f'birth_modified_{i}', f'death_modified_{i}']
    colnames += ['Hausdorff_distance']
    colnames += ['dropped_loci']
    writer.writerow(colnames)
    # exp_results = []

    for ag_only in [True, False]:
        for n_drop in range(config.drop_min, config.drop_max+config.drop_step, config.drop_step):
            if n_drop > np.sum(mask_AG_GA) and ag_only:
                continue
            print('\n', n_drop)
            # n_drop = 20
            rng = np.random.default_rng(42)
            splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_iter, random_state=42).split(X_oe, y)
            for num in range(n_iter):
                print(num, end=' ')
                col_drop = np.full((fd.shape[0], ), False)
                col_drop[rng.choice(
                    (np.where(mask_AG_GA)[0] if ag_only else np.arange(fd.shape[0])), 
                    size=n_drop, replace=False)] = True
                dropped_loci = ','.join(list(df.columns[3:][col_drop]))
                models = [
                    ('pca+l2', make_pipeline(
                        OneHotEncoder(
                            handle_unknown='ignore',
                            sparse=False,
                        ),
                        PCA(
                            n_components=30,
                            random_state=42
                        ),
                        LogisticRegression(
                            max_iter=1000,
                            random_state=42
                        )
                    )),
                    ('pca+l1', make_pipeline(
                        OneHotEncoder(
                            handle_unknown='ignore',
                            sparse=False,
                        ),
                        PCA(
                            n_components=30,
                            random_state=42
                        ),
                        LogisticRegression(
                            solver='liblinear',
                            penalty='l1',
                            max_iter=1000,
                            random_state=42
                        )
                    )),
                    ('pca+DT', make_pipeline(
                        OneHotEncoder(
                            handle_unknown='ignore',
                            sparse=False,
                        ),
                        PCA(
                            n_components=30,
                            random_state=42
                        ),
                        DecisionTreeClassifier(
                            max_depth=4,
                            criterion='entropy',
                            random_state=42
                        )
                    )),
                    ('pca+GM', make_pipeline(
                        OneHotEncoder(
                            handle_unknown='ignore',
                            sparse=False,
                        ),
                        PCA(
                            n_components=5,
                            random_state=42
                        ),
                        GaussianMixturePP(
                            n_components=3,
                            random_state=42
                        )
                    )),
                    ('CatNB', make_pipeline(
                        CategoricalNB(
                            min_categories=min_categories
                        )
                    ))
                ]
                for (train_idxs, test_idxs), _ in zip(splitter, range(5)):
                    if ag_only:
                        dist_control = np.zeros_like(dist_f)
                        for i, j in np.ndindex(dist_control.shape):
                            if i >= j: continue
                            dist_control[i, j] = np.mean(fd.iloc[i, train_idxs] != fd.iloc[j, train_idxs])
                            dist_control[j, i] = dist_control[i, j]
                        dist_modified = dist_control[mask_AG_GA&(~col_drop)][:, mask_AG_GA&(~col_drop)]
                        dist_control = dist_control[mask_AG_GA][:, mask_AG_GA]
                        hausdorff_dist = np.max(np.min(dist_control[col_drop[mask_AG_GA]][:, ~col_drop[mask_AG_GA]], axis=1), axis=0)
                        # pers_control = _extract_persistence_from_data(
                        #     X_oe[train_idxs].T[mask_AG_GA], 
                        #     metric='hamming', top_k=top_k)
                        # pers_modified = _extract_persistence_from_data(
                        #     X_oe[train_idxs].T[mask_AG_GA & (~col_drop)], 
                        #     metric='hamming', top_k=top_k)
                    else:
                        base3_train = base3_with_perm[:, :, train_idxs]
                        dist_control = np.zeros_like(dist_q)
                        for i, j in np.ndindex(dist_control.shape):
                            if i >= j: continue
                            dist_control[i, j] = np.min(np.mean(base3_train[0, i] != base3_train[:, j], axis=-1))
                            dist_control[j, i] = dist_control[i, j]
                        dist_modified = dist_control[~col_drop][:, ~col_drop]
                        hausdorff_dist = np.max(np.min(dist_control[col_drop][:, ~col_drop], axis=1), axis=0)
                        # pers_control = _extract_persistence_from_data(dist_control, metric='precomputed', top_k=top_k)

                        # dist_modified = dist_control[~col_drop][:, ~col_drop]
                        # pers_modified = _extract_persistence_from_data(dist_modified, metric='precomputed', top_k=top_k)
                    pers_control = _extract_persistence_from_data(dist_control, metric='precomputed', top_k=top_k)
                    pers_modified = _extract_persistence_from_data(dist_modified, metric='precomputed', top_k=top_k)


                    for model_name, model in models:
                        model.fit(X_oe[train_idxs], y[train_idxs])
                        acc_control = 1e2*accuracy_score(y[test_idxs], model.predict(X_oe[test_idxs]))

                        final_estimator = model._final_estimator
                        if hasattr(final_estimator, 'min_categories'):
                            final_estimator.set_params(min_categories=min_categories[~col_drop])

                        model.fit(X_oe[train_idxs][:, ~col_drop], y[train_idxs])
                        acc_modified = 1e2*accuracy_score(y[test_idxs], model.predict(X_oe[test_idxs][:, ~col_drop]))

                        if hasattr(final_estimator, 'min_categories'):
                            final_estimator.set_params(min_categories=min_categories)

                        exp_result = np.concatenate([
                            [[acc_control, acc_modified]],
                            pers_control, pers_modified,
                        ], axis=0)
                        exp_result = [ag_only, n_drop, num, model_name] + exp_result.ravel().tolist()
                        exp_result += [hausdorff_dist]
                        exp_result += [dropped_loci]
                        # exp_results.append(exp_result)

                        writer.writerow(exp_result)

    # for row in exp_results:
    #     writer.writerow(row)
    file_w.close()
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)