import datautils
import numpy as np
import os

class ReduceDict:
    # like reduce() but with dict to cumulate by key
    def __init__(self, start, operation):
        # operation is function(cummulate, value, key)
        self.start = start
        self.operation = operation
        self.cumulate = dict()
        self.n = dict()

    def keys(self):
        return self.cumulate.keys()

    def iter(self, key, value):
        self.cumulate[key] = self.operation(self.cumulate.get(key, self.start), value, key)
        self.n[key] = self.n.get(key, 0) + 1


def feature_extractor(row):
    # split features_code and features from "features" column
    features = [int(el) for el in row['features'].split(',')]
    features_code = features[0]
    features = np.array(features[1: ])
    return features_code, features

def np_divide_zero(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def main(root):
    num_features = 256
    float_type = np.float64
    train_tsv = datautils.TsvFileReader(os.path.join(root, 'train.tsv'))

    # sum(x)
    train_tsv.open()
    sums1 = ReduceDict( np.zeros( (num_features,) ), lambda c, v, k: c + v)
    for i, row in enumerate(train_tsv.iterrows()):
        try:
            features_code, features = feature_extractor(row)
        except ValueError:
            raise ValueError('Bad line {} in {}: "{}"'.format(i, train_tsv.path, row['features']))
        sums1.iter(features_code, features)
    train_tsv.close()

    # mean_x
    means = dict()
    for features_code in sums1.keys():
        means[features_code] = float_type(sums1.cumulate[features_code]) / float_type([sums1.n[features_code]])

    # sum((x - mean_x) ** 2)
    train_tsv.open()
    sums2_center = ReduceDict( np.zeros( (num_features,) ), lambda c, v, k: c + ((v - means[k]) ** 2))
    for row in train_tsv.iterrows():
        features_code, features = feature_extractor(row)
        sums2_center.iter(features_code, features)
    train_tsv.close()

    # sigma_x
    sigmas = dict()
    for features_code in sums2_center.keys():
        sigmas[features_code] = np.sqrt(sums2_center.cumulate[features_code] / float_type(sums2_center.n[features_code]))

    # test_proc calculations
    test_tsv = datautils.TsvFileReader(os.path.join(root, 'test.tsv'))
    testproc_columns = ['id_job', 'feature_stand', 'max_feature_index', 'max_feature_abs_mean_diff']
    testproc_tsv = datautils.TsvFileWriter(os.path.join(root, 'test_proc.tsv'), testproc_columns)

    test_tsv.open()
    testproc_tsv.open()
    for row in test_tsv.iterrows():
        features_code, features = feature_extractor(row)
        new_row = dict()
        new_row['id_job'] = row['id_job']
        z_norm = np_divide_zero(float_type(features) - means[features_code], sigmas[features_code])
        new_row['feature_stand'] = ','.join(str(el) for el in z_norm)
        max_feature_index = np.argmax(features)
        new_row['max_feature_index'] = max_feature_index
        new_row['max_feature_abs_mean_diff'] = np.abs(
            float_type(features[max_feature_index]) - means[features_code][max_feature_index])
        testproc_tsv.write_row(new_row)
    test_tsv.close()
    testproc_tsv.close()


if __name__ == "__main__":
    main(root = '.')
