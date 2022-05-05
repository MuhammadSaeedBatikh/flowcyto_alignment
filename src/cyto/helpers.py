import numpy as np
from scipy.stats import  wasserstein_distance



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
