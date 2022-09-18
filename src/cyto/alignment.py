
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import copy
import statsmodels.api as sm
import scipy
# from pyearth import Earth
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge


def minmaxmean_align(x,y, mi, mx, q_alignment, func0 = None):

    '''

    Creates a warping function that soft aligns target boundaries and mean oto corresponding reference values.

    :param x: :obj:`numpy.array(dtype=float)`, shape = [1, n], A 1-D array containing target data.
    :param y: :obj:`numpy.array(dtype=float)`, shape = [1, n], A 1-D array containing reference data.
    :param mi: :obj:`float`, reference segment left quantile.
    :param mx: :obj:`float`, reference segment right quantile.
    :param q_alignment: :obj:`float`. A quantile value for robust alignment. Instead of matching min and max values of reference, the function matches q_alignment and 1-q_alignment quantiles.
    :param func0: :obj:`function, the transformation that needs to be applied before the minmax_alignment, typically the inverse probability transform.
    :return: :obj:`function`, a warping function that aligns the boundaries of target to reference gates.
    '''

    #  f(x)   = aa x**3 + alpha * x**2 + beta*x + gamma
    #  f(x)   = alpha * x**2 + beta*x + gamma
    # E(f(x)) = alpha * Ex*2 + beta*EY + gamma = EY
    # f(x_mx) = alpha * x_mx*2 + beta*x_mx + gamma = y_mx
    # f(x_mi) = alpha * x_mi*2 + beta*x_mi + gamma = y_mi

    # TODO: ensure monotonicity. Fix bug with quadratic warping.

    if func0 is None:
        func0 = lambda x:x

    z = func0(x)
    ind = ~np.isnan(z)
    x_mean, y_mean =z[ind].mean(), y.mean()

    # TODO: use the precomputed quantiles for target gate
    x_mx, x_mi = np.quantile(z[ind],1-q_alignment), np.quantile(z[ind], q_alignment)
    M = np.array([
        [x_mx**2, x_mx, 1],
        [x_mi**2, x_mi, 1],
        [z[ind].std()**2 + x_mean**2, x_mean, 1]
    ])
    b = np.array([mx, mi, y_mean])
    s = np.linalg.inv(M)@b.T
    alpha, beta, gamma = s[0], s[1], s[2]
    print('alpha',alpha, 'beta',beta, 'gamma',gamma)
    return lambda x: alpha * func0(x)**2 + beta * func0(x) + gamma

def linear_reg_inter(ymean, xmean, x0, y0):

    '''

    :param ymean:
    :param xmean:
    :param x0:
    :param y0:
    :return:
    '''

    a = (ymean - y0)/(xmean-x0)
    f = lambda x: a*x + y0-a*x0
    return f

# def comb_func_factory(x, f_start, f_middle, f_end, x_cut_point_start, x_cut_point_end):
#     ind_st = x<x_cut_point_start
#     ind_end = x>x_cut_point_end
#     ys = f_start(x[ind_st])
#     ym = f_middle(x[~np.logical_or(ind_st, ind_end)])
#     ye = f_end(x[ind_end])
#     return np.concatenate([ys, ym, ye])
#
#
# def combine_funcs(funcs, locations, smooth_sigma=0.1):
#     xcom = []
#     ycom = []
#     plot = True
#     for i, f in enumerate(funcs):
#         g0, g1 = locations[i]
#         print(g0, g1)
#         x_s = np.linspace(g0, g1, 2000)
#         y_s = f(x_s)
#         xcom += [x_s]
#         ycom += [y_s]
#         if plot:
#             plt.plot(x_s, y_s)
#             plt.xlim(-.3,1.3)
#             plt.ylim(-.3,1.3)
#     xcom = np.concatenate(xcom)
#     ycom = np.concatenate(ycom)
#     # middle
#     f_middle = scipy.interpolate.interp1d(xcom, ycom, kind='cubic')
#
#     # start
#     x_cut_point_end = locations[0][0] + .1*np.abs(locations[0][0]-locations[0][1])
#     ind = np.argmin(np.abs(xcom-x_cut_point_end))
#     x_cut_point_start = xcom[ind]
#     y_cut_point_start = ycom[ind]
#     ind = xcom<=x_cut_point_start
#     x, y = xcom[ind], ycom[ind]
#     f_start = linear_reg_inter(y.mean(), x.mean(), x_cut_point_start, y_cut_point_start)
#     x = np.linspace(-.3, x_cut_point_start, 20000)
#     if plot:
#         plt.figure()
#         plt.plot(x, f_start(x))
#
#     # end
#     x_cut_point_end = locations[-1][1] - .1*np.abs(locations[-1][0]-locations[-1][1])
#     ind = np.argmin(np.abs(xcom-x_cut_point_end))
#     x_cut_point_end = xcom[ind]
#     y_cut_point_end = ycom[ind]
#     ind = xcom >= x_cut_point_end
#     x, y = xcom[ind], ycom[ind]
#     f_end = linear_reg_inter(y.mean(), x.mean(), x_cut_point_end, y_cut_point_end)
#     if plot:
#         x = np.linspace(x_cut_point_end, 1.3, 20000)
#         plt.plot(x, f_end(x))
#         x = np.linspace(x_cut_point_start, x_cut_point_end, 20000)
#         plt.plot(x, f_middle(x))
#
#     comb_func = lambda x: gaussian_filter1d(comb_func_factory(x, f_start, f_middle, f_end,
#                                                               x_cut_point_start, x_cut_point_end),smooth_sigma)
#     return comb_func
#

def lambda_IPT_factory(ref_inv_cdf, target_ecdf):

    '''
    A factory method for Inverse Probability Transform. The reason why it is necessary to have this kind of factory/additional layer is that
    python lambdas do not store default hyperparamters for non-primitive data types and will access the most recent parameters in its closure.
    By having a factory, we ensure that parameters are snapshotted in the scope of the factory and the returned lambda closure has its own unique hyperparameters.

    :param ref_inv_cdf: :obj:`function`, inverse empirical cumulative distribution function (ECDF) of reference segment
    :param target_ecdf: :obj:`function`, ECDF of target segment.
    :return: :obj:`function`, inverse probability transform.
    '''

    return lambda x: ref_inv_cdf(target_ecdf(x))

def compute_cdf_and_inv_cdf(ref_gate, k=3):

    '''

    Computes the CDF and Inverse CDF of a gate. (typically for a reference gate)

    :param ref_gate: :obj:`Gate`, reference gate.
    :param k: :obj:`int`, BSpline degree (default =3).
    :returns:
         - ref_ecdf - :obj:`function`, reference Empirical Cumulative distribution function.
         - ref_inv_cdf - :obj:`function`, reference inverse Cumulative distribution function.

    '''

    ref_seg = ref_gate.segment
    # Compute empirical cumulative distribution function for the reference segment
    ref_ecdf = sm.distributions.empirical_distribution.ECDF(ref_seg)
    # Compute the inverse of the ECDF of reference using monotone function inverter
    ref_inv_cdf = sm.distributions.empirical_distribution.monotone_fn_inverter(ref_ecdf,
                                                                               ref_seg)
    # fit a BSpline tp inverse ECDF

    ref_inv_cdf = scipy.interpolate.BSpline(
        ref_inv_cdf.x,
        ref_inv_cdf.y,
        k=k,
        extrapolate=False
    )
    return ref_ecdf, ref_inv_cdf

def get_smooth_and_monotonic(t, y):

    '''

    A convolution-based transform to ensure smoothness and monotonicity.

    :param t:
    :param y:
    :return:
    '''

    N = y.shape[0]//10
    yy =  np.convolve(y, np.ones(N)/N, mode='valid')
    yy, idx = np.unique(yy, return_index=True)
    tt = t[idx]
    return np.sort(tt), np.sort(yy), [tt.min(), tt.max()]

def combined(x, model, funcs, locations, sigma, only_use_MARS = False ):

    '''

    The final combined function of MARS model and segments transformations.

    :param x: :obj:`numpy.array(dtype=float)`, shape = [1, n], A 1-D array containing target data.
    :param model: model: :obj:`pyearth.Earth` MARS model.
    :param funcs: obj:`list(function)`, a list of warping functions for each segment.
    :param locations: obj:`list(tuple)`, a list of tuples that contain the start and end range of each function.
    :return: y: :obj:`numpy.array(dtype=float)`, shape = [1, n], A 1-D array containing transformed target data.
    '''

    if only_use_MARS:
        y = model.predict(x)
        return y

    y = np.zeros(x.shape)
    all_inds = []
    for i, g in enumerate(locations):
        ind = np.logical_and(x>=g[0], x<g[1])
        f = funcs[i]
        if f is not None:
            y[ind] =  gaussian_filter1d( f(x[ind]), sigma )
            all_inds+=[ind]
    if len(all_inds)>1:
        remaining_ind = ~np.logical_or(*all_inds)
    else:
        remaining_ind = ~all_inds[0]

    if np.count_nonzero(remaining_ind)>0:
        y[remaining_ind] = model.predict(x[remaining_ind])
    return y

def combine_funcs(model, xs_ys, funcs, locations, sigma, subsample_ratio=1, plot = True, ax = None):

    '''

    Creates a combined function of MARS model and every segments transformations.

    :param model: :obj:`pyearth.Earth` MARS model.
    :param xs_ys: obj:`list(list(numpy.array))`, a list of x values and transformed y values in the form of [ [target_seg1, f(target_seg1)], [target_seg2, f(target_seg2)], ...].
    :param funcs: obj:`list(function)`, a list of warping functions for each segment.
    :param locations: obj:`list(tuple)`, a list of tuples that contain the start and end range of each function.
    :param sigma: :obj:`float`, the standard deviation of Gaussian kernel.
    :param subsample_ratio: :obj:`float`, a number between [0,1]`, represents the ratio of cells to consider to estimate MARS parameters.
    :param plot: :obj:`Bool`, if :obj:`True`, plots warping functions.
    :param ax: :obj:`matplotlib.axes.Axes`, axes to draw on. If :obj:`None`, a new axes is created.
    :return: :obj:`function`, the complete warping function for a sample's channel including its MARS regression model for extrapolation.
    '''
    xcom = []
    ycom = []

    if plot and ax is None:
        ax = plt.subplots(1,1)[1]
    for i, xy in enumerate(xs_ys):
        x_s, y_s,= xs_ys[i][0],xs_ys[i][1]
        if 0<subsample_ratio<1:
            m = x_s.shape[0]
            indx = np.random.randint(0, m, int(m*subsample_ratio))
            xcom += [x_s[indx]]
            ycom += [y_s[indx]]
        elif subsample_ratio == 1:
            xcom += [x_s]
            ycom += [y_s]
        else:
            raise Exception(f'subsample_ratio must be >0 and <=1, you passed {subsample_ratio}')
        if plot:
            ax.plot(x_s, y_s, linewidth=3)
    xcom = np.concatenate(xcom)
    ycom = np.concatenate(ycom)
    # middle
    #Must subsample for  for Kernel Ridge CV
    # Must subsample for  for Kernel Ridge CV
    m = xcom.shape[0]
    indx = np.random.randint(0, m, np.minimum(m, 1000))
    model.fit(xcom[indx].reshape(-1, 1), ycom[indx])

    comb_func = lambda x: combined(x, model, funcs, locations, sigma)
    if plot:
            t = np.linspace(-.1,1.1,5000)
            ax.plot(t, comb_func(t))
            ax.set_xlim(-.3,1.3)
            ax.set_ylim(-.3,1.3)
    return comb_func



def estimate_alignment_funcs_for_target(ch, target_sample, q_alignment, Loc_Ref_Dict_All_Ch, Ref_Dict_indx_by_Loc_and_Morph, Ref_CDF_and_InvCDF, verbose):


    '''

    Estimates transformations/functions for each gate to align target_sample to provided references.

    :param ch: :obj:`int`, channel number
    :param target_sample:  :obj:`Sample`. target sample.
    :param q_alignment: :obj:`float`. A quantile value for robust alignment. Instead of matching min and max values of reference, the function matches q_alignment and 1-q_alignment quantiles.
    :param Loc_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of :obj:`Gate` chosen reference gates for all channels with channels numbers as its keys.
    :param Loc_Morph_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of dictionaries of :obj:`Gate` chosen morphology reference gates for all channels with (channel number, location group) tupel as its keys.
    :param Ref_CDF_and_InvCDF: :obj:`dict`, a dictionary of :obj:`tuple(function, function), of reference gates ecdf and inverse cdf respectively.
    :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel in sample.
    :returns:
         - funcs - :obj:`list(functions)`, a list of estimated funcs per gate. If no reference was found for a particular gate, a None is put in place of to be replaced by MARS.
         - xys - :obj:`list(tuple)`, a list of tuples (x, y), where x is a 1D array representing the input value of each gate, and y is its estimated (transformed) counterpart.
         - gates_locations - :obj:`list(tuple)`, a list of tuples (left boundary, right boundary).
         - sample_orignal - :obj:`numpy.array`, original sample.

    '''

    funcs = []
    z = target_sample(ch)
    sample_orignal = copy.deepcopy(z)
    gates_locations = []
    xys = []

    for i in range(len(target_sample.gates[ch])):
        target_gate = target_sample.gates[ch][i]
        ind = np.logical_and(z >= target_gate.gate[0], z < target_gate.gate[1])
        target_seg1 = target_gate.segment
        lo_gr, mr_gr = target_gate.location_group, target_gate.morphology_group

        ref_gate = Loc_Ref_Dict_All_Ch[ch][lo_gr]
        mr_gr_ref = Ref_Dict_indx_by_Loc_and_Morph.get((lo_gr, mr_gr))


        if verbose:
            print('diff:', np.abs(target_gate.segment - target_seg1).sum())
            print('lo_gr, mr_gr', lo_gr, mr_gr)
            print('\n mr_gr_ref', mr_gr_ref, '\n')
            print('target gate', target_gate, '\n')

        if target_gate.leave_to_MARS:
            funcf = None

        else:
            if mr_gr_ref:
                ref_cdf, ref_inv_cdf = Ref_CDF_and_InvCDF.get((lo_gr, mr_gr))

                if mr_gr_ref != target_gate:
                    # Compute empirical cumulative distribution function for the target segment
                    target_ecdf = sm.distributions.empirical_distribution.ECDF(target_seg1)
                    func0 = lambda_IPT_factory(ref_inv_cdf, target_ecdf)

                else:
                    func0 = lambda x: x

            q0, q1 = ref_gate.get_tight_gates(q=q_alignment)
            if mr_gr_ref:
                funcf = minmaxmean_align(target_seg1, ref_gate.segment, q0, q1, q_alignment, func0)

            else:
                funcf = minmaxmean_align(target_seg1, ref_gate.segment, q0, q1, q_alignment)

        funcs += [funcf]
        if verbose:
            print('loc. gate is alredy ref?', ref_gate == target_gate)
            print('gate:\n', target_gate.gate)
        sample_orignal[ind] = z[ind]
        if not target_gate.leave_to_MARS:
            t = np.linspace(target_gate.tight_gate[0], target_gate.tight_gate[1], 2000)
            y = funcf(t)
            ind = ~np.isnan(y)
            xys += [[t[ind], y[ind]]]
        gates_locations += [[target_gate.gate[0], target_gate.gate[1]]]

    return funcs, xys, gates_locations, sample_orignal


def align_samples_func(ch, q_alignment,
                  samples,
                  aligned_samples,
                  original_samples,
                  Loc_Ref_Dict_All_Ch,
                  Loc_Morph_Ref_Dict_All_Ch,
                  Ref_Inv_CDF_Dict_All_Ch,
                  funcs_dict, comp_func_dict,earth_models_dict,
                  gates_locations_dict,
                  n_sample,
                  sigma, earth_smoothing_penalty, subsample_ratio=1,
                  in_place_eval = True,
                  verbose=False):
    '''

    Transforms channel ch unaligned data to aligned data based on the provided references and fills in the provided aligned samples dictionary.

    :param ch: :obj:`int`, channel number.
    :param q_alignment: :obj:`float`. A quantile value for robust alignment. Instead of matching min and max values of reference, the function matches q_alignment and 1-q_alignment quantiles.
    :param samples: :obj:`list(Sample)`, a list of all samples in data as :obj:`Sample` objects.
    :param aligned_samples: :obj:`dict`, a dictionary of :obj:`numpy.array(dtype=float)` aligned_samples,  with its keys is represented as the tupel (sample number, channel).
    :param original_samples: :obj:`dict`, a dictionary of :obj:`numpy.array(dtype=float)` original_samples, morphology groups for each channel with channels numbers as its keys.
    :param Loc_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of :obj:`Gate` chosen reference gates for all channels with channels numbers as its keys.
    :param Loc_Morph_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of dictionaries of :obj:`Gate` chosen morphology reference gates for all channels with (channel number, location group) tupel as its keys.
    :param funcs_dict: :obj:`dict`, a dictionary of :obj:`list` warping functions for each gate in a channel of a sample, with its keys is represented as the tupel (sample number, channel).
    :param comp_func_dict: :obj:`dict`, a dictionary of :obj:`function` combined warping functions for each channel in each sample, with its keys is represented as the tupel (sample number, channel).
    :param Ref_Inv_CDF_Dict_All_Ch: :obj:`dict`, a dictionary of dictionaries of :obj:`tuple(function, function) of reference gates ecdf and inverse cdf respectively.
    :param earth_models_dict: :obj:`dict`, a dictionary of :obj:`pyearth.Earth` MARS models for each channel in each sample, with its keys is represented as the tupel (sample number, channel).
    :param gates_locations_dict: :obj:`dict`, a dictionary of :obj:`list` location of gates,  with its keys is represented as the tupel (sample number, channel).
    :param n_sample: obj:`int`, number os samples to align.
    :param sigma: :obj:`float`, the standard deviation of Gaussian kernel.
    :param earth_smoothing_penalty: :obj:`float`, A smoothing parameter used to calculate generalized cross validation. Used during the pruning pass and to determine whether to add a hinge or linear basis function during the forward pass.
    :param subsample_ratio: :obj:`float`, a number between [0,1]`, represents the ratio of cells to consider to estimate MARS parameters.
    :param in_place_eval: :obj:`Bool`, if :obj:`True`, aligns data within.
    :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel in sample.
    :returns:
         - comp_func_dict - :obj:`dict`, a dictionary of :obj:`function` combined warping functions for each channel in each sample, with its keys is represented as the tupel (sample number, channel).
         - funcs_dict - :obj:`dict`, a dictionary of :obj:`list` warping functions for each gate in a channel of a sample, with its keys is represented as the tupel (sample number, channel).
         - earth_models_dict - :obj:`dict`, a dictionary of :obj:`pyearth.Earth` MARS models for each channel in each sample, with its keys is represented as the tupel (sample number, channel).

    '''

    Ref_Dict_indx_by_Loc_and_Morph = Loc_Morph_Ref_Dict_All_Ch[ch]
    Ref_CDF_and_InvCDF = {}
    for (lo_gr, mr_gr), ref_gate in Ref_Dict_indx_by_Loc_and_Morph.items():
        ref_ecdf, ref_inv_cdf = compute_cdf_and_inv_cdf(ref_gate)
        Ref_CDF_and_InvCDF[lo_gr, mr_gr] = (ref_ecdf, ref_inv_cdf)

    Ref_Inv_CDF_Dict_All_Ch[ch] = Ref_CDF_and_InvCDF

    for targ_num in range(n_sample):
        if verbose:
            print(f'Sample {targ_num} **** \n')

        target_sample = samples[targ_num]
        funcs, xys, gates_locations, sample_orignal = estimate_alignment_funcs_for_target(ch,target_sample= target_sample, q_alignment=q_alignment,
                                                                       Loc_Ref_Dict_All_Ch= Loc_Ref_Dict_All_Ch,
                                                                       Ref_Dict_indx_by_Loc_and_Morph=Ref_Dict_indx_by_Loc_and_Morph,
                                                                       Ref_CDF_and_InvCDF=Ref_CDF_and_InvCDF, verbose= verbose )

        model = GridSearchCV(
            KernelRidge(kernel="rbf", gamma=0.1, alpha=earth_smoothing_penalty),
            param_grid={"alpha": [1, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 9)},
        )

        # model = Earth(penalty=earth_smoothing_penalty, smooth=True, max_degree=1, )
        comp_func = combine_funcs(model, xys, funcs, gates_locations, sigma=sigma, subsample_ratio=subsample_ratio, plot=False)
        comp_func_dict[ch, targ_num] = comp_func
        earth_models_dict[ch, targ_num] = model
        funcs_dict[targ_num, ch] = funcs

        if in_place_eval:
            gates_locations_dict[targ_num, ch] = gates_locations
            sample_aligned = comp_func(sample_orignal)
            aligned_samples[targ_num, ch] = sample_aligned
            original_samples[targ_num, ch] = sample_orignal

    return comp_func_dict, funcs_dict, earth_models_dict


