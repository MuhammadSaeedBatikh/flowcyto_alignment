import numpy as np
from scipy.stats import  wasserstein_distance
import pandas as pd
import flowio
import os




def load_data(path, unique_samples_codes, sample_code_column, sep=','):

    '''

    Loads csv data file from disk. First row of file (header) indicates markers. First cell of first row indicates sample code.

    :param path: :obj:`Str`, data path.
    :param sample_code_column: :obj:`Str`, name of the column of sample codes/ labels.
    :param sep: :obj:`Str`, data separator.
    :returns:
         - X - :obj:`numpy.array(float)`, an n*m cells matrix where n is the number of cells, and m number of markers. (not including cells codes/labels )
         - Y - :obj:`numpy.array(int)`, an n*1 integer labels array indicating sample number of each cell, where n is the number of cells.
         - columns_names - :obj:`list(str)`, columns names (header) including cells codes/labels.
    '''

    df = pd.read_csv(path, sep=sep)
    columns_names = df.columns.tolist()
    # samples_codes = df[sample_code_column]
    # samples_codes = np.unique(samples_codes)
    # samples_codes = np.unique([e.split('-')[0]for e in samples_codes])
    # Todo: check for a standard to find unique labels

    X = []
    Y = []

    for i, s in enumerate(unique_samples_codes):
        sample_train = df[df[sample_code_column].str.contains(s)].to_numpy()[:,1:].astype(np.float)
        X += [sample_train]
        y = i*np.ones(shape= sample_train.shape[0])
        Y+=[y]

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    return X, Y, columns_names, None


def load_data_from_a_list_of_matrices(samples_list, columns_names):

    '''

       Loads data from a list of samples. Each sample is an m*n matrix.

       :param samples_list: :obj:`list`, list of matrices.
       :param columns_names :obj:`list(str)`, columns names (header) including cells codes/labels.
       :returns:
            - X - :obj:`numpy.array(float)`, an n*m cells matrix where n is the number of cells, and m number of markers. (not including cells codes/labels )
            - Y - :obj:`numpy.array(int)`, an n*1 integer labels array indicating sample number of each cell, where n is the number of cells.
            - columns_names - :obj:`list(str)`, columns names (header) including cells codes/labels.

    '''

    Y = []
    for i, s in enumerate(samples_list):
        y = i * np.ones(shape=s.shape[0])
        Y += [y]

    X = np.concatenate(samples_list)
    Y = np.concatenate(Y)
    return X, Y, columns_names, None


def load_data_from_dictionary(samples_dict, columns_names):
    files = list(samples_dict.keys())
    samples_list = list(samples_dict.values())
    X, Y, columns_names, _ = load_data_from_a_list_of_matrices(samples_list, columns_names)
    files_dict = {y:v for y, v in enumerate(files)}
    return X, Y, columns_names, files_dict


def load_data_from_files(root_folder, format ='fcs', sep=','):
    '''

    :param root_folder: :obj:`str`
    :param format: :obj:`str`, supported formats ('fcs' or 'csv')
    :param sep: :obj:`str`, separation used for csv file
-   :returns:
         - X - :obj:`numpy.array`, data array.
         - Y - :obj:`numpy.array`, an array representing samples numbers.
         - columns_names - :obj:`list`, list of gates in channel ch.
         - files_dict - :obj:`dict`, dictionary of samples numbers (key) and files names (value).

    '''


    files = list(os.walk(root_folder))[0][2]

    # only consider files in directory with format (.format)
    files = list(filter(lambda x: format.lower() in x, files))

    if len(files) <1:
        raise Exception(f'no files found in folder {root_folder}')

    samples = []
    Y = []
    for y, path in enumerate(files):

        if format.lower() =='fcs':
            flow_data = flowio.FlowData(os.path.join(root_folder, path))
            X = np.array(flow_data.events).reshape(-1, flow_data.channel_count)
        elif format.lower() =='csv':
            df = pd.read_csv(os.path.join(root_folder, path), sep=sep)
            X = df.to_numpy()
        else:
            raise ValueError(f'No support for format: {format}')

        y = np.ones(X.shape[0], dtype=np.int)*int(y)
        Y += [y]
        samples+=[X]

    if format.lower() =='fcs':
        columns_names = [list(v.values())[0] for v in flow_data.channels.values()]
    elif format.lower() =='csv':
        columns_names = df.columns.tolist()

    Y = np.concatenate(Y, 0)
    X = np.concatenate(samples, 0)

    files_dict = {y:v for y, v in enumerate(files)}
    return X, Y, columns_names, files_dict



def jaccard(A, B):

    '''

    Computes the Jaccard similarity index (overlap/union) between two line segments.
    For example A=[0.25,0.5], B=[0.3,0.75], jaccard(A, B) returns (0.5-0.3)/(0.75-0.25)

    :param A: :obj:`list`: a list with two elements that represent the start and end of a line segment.
    :param B: :obj:`list`: a list with two elements that represent the start and end of a line segment.
    :returns:
     - jacc_indx - :obj:`float`, Jaccard similarity index.
    '''

    a = np.min([np.min(A),np.min(B)])
    b = np.max([np.max(A),np.max(B)])
    union = np.abs((b-a))
    if np.max(A) <= np.min(B) or np.max(B) <= np.min(A):
        overlap = 0
    else:
        a = np.max([np.min(A),np.min(B)])
        b = np.min([np.max(A),np.max(B)])
        overlap = np.abs((b-a))
    jacc_indx = overlap/union
    return jacc_indx


def weighted_jaccard(A, B, z):

    '''

    Computes a weighted version of Jaccard similarity index (weight * overlap/union) between two line segments.

    :param A: :obj:`list`: a list with two elements that represent the start and end of a line segment.
    :param B: :obj:`list`: a list with two elements that represent the start and end of a line segment.
    :param z: :obj:`numpy.array`, shape = [m, 1] where m is the number of cells.
    :returns:
      - jacc_indx - :obj:`float`, weighted Jaccard similarity index.
    '''

    a = np.min([np.min(A),np.min(B)])
    b = np.max([np.max(A),np.max(B)])
    union = np.abs((b-a))
    a = np.max([np.min(A),np.min(B)])
    b = np.min([np.max(A),np.max(B)])
    overlap = np.abs((b-a))
    j = overlap/union
    ind = np.logical_and(z>=a, z<=b)
    w = z[ind].shape[0]/z.shape[0]
    return j*w



def morphology_distance(z1, z2):
    '''

    Computes the Wasserstein Distance after the standardization of the two curves.
    The result is raised to a high power (dist^8) in order to amplify distributional differences.

    :param z1: :obj:`numpy.array`, shape = [m, 1] where m is the number of cells.
    :param z2: :obj:`numpy.array`, shape = [n, 1] where m is the number of cells.
    :return: :obj:`float` Wasserstein distance.
    '''

    dist = wasserstein_distance((z1-z1.mean())/z1.std(),
                                (z2-z2.mean())/z2.std()
                                )
    dist = np.power(dist, 8)
    return dist
