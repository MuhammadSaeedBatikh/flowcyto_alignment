
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import copy
import statsmodels.api as sm
import scipy

def minmaxmean_align(x,y, mi, mx, q_alignment, func0 = None):

    '''

    :param x:
    :param y:
    :param mi:
    :param mx:
    :param q_alignment:
    :param func0:
    :return:
    '''

    #  f(x)   = aa x**3 + alpha * x**2 + beta*x + gamma
    #  f(x)   = alpha * x**2 + beta*x + gamma
    # E(f(x)) = alpha * Ex*2 + beta*EY + gamma = EY
    # f(x_mx) = alpha * x_mx*2 + beta*x_mx + gamma = y_mx
    # f(x_mi) = alpha * x_mi*2 + beta*x_mi + gamma = y_mi

    if func0 is None:
        func0 = lambda x:x

    z = func0(x)
    ind = ~np.isnan(z)
    x_mean, y_mean =z[ind].mean(), y.mean()
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
from pyearth import Earth

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

def lambda_warp_factory(ref_inv_cdf, target_ecdf):

    '''

    :param ref_inv_cdf:
    :param target_ecdf:
    :return:
    '''

    return lambda x:ref_inv_cdf(target_ecdf(x))

def get_smooth_and_monotonic(t, y):

    '''

    :param t:
    :param y:
    :return:
    '''

    N = y.shape[0]//10
    yy =  np.convolve(y, np.ones(N)/N, mode='valid')
    yy, idx = np.unique(yy, return_index=True)
    tt = t[idx]
    return np.sort(tt), np.sort(yy), [tt.min(), tt.max()]

def combined(x, model, funcs, locations):

    '''

    :param x:
    :param model:
    :param funcs:
    :param locations:
    :return:
    '''

    y = np.zeros(x.shape)
    all_inds = []
    for i, g in enumerate(locations):
        ind = np.logical_and(x>=g[0], x<g[1])
        f = funcs[i]
        y[ind] =  gaussian_filter1d( f(x[ind]), 1)
        all_inds+=[ind]
    if len(all_inds)>1:
        remaining_ind = ~np.logical_or(*all_inds)
    else:
        remaining_ind = ~all_inds[0]

    if np.count_nonzero(remaining_ind)>0:
        y[remaining_ind] = model.predict(x[remaining_ind])
    return y

def combine_funcs(model, xs_ys, funcs, locations, plot = True, ax = None):

    '''

    :param model:
    :param xs_ys:
    :param funcs:
    :param locations:
    :param plot:
    :param ax:
    :return:
    '''

    xcom = []
    ycom = []
    if plot and ax is None:
        ax = plt.subplots(1,1)[1]
    for i, xy in enumerate(xs_ys):
        print(i)
        # x_s,y_s,_ = get_smooth_and_monotonic(xs_ys[i][0],xs_ys[i][1])
        x_s,y_s,= xs_ys[i][0],xs_ys[i][1]
        xcom += [x_s]
        ycom += [y_s]
        if plot:
            ax.plot(x_s, y_s, linewidth=3)
    xcom = np.concatenate(xcom)
    ycom = np.concatenate(ycom)
    # middle
    model.fit(xcom, ycom)
    comb_func = lambda x: combined(x, model, funcs, locations)
    if plot:
            t = np.linspace(-.1,1.1,5000)
            ax.plot(t, comb_func(t))
            ax.set_xlim(-.3,1.3)
            ax.set_ylim(-.3,1.3)
    return comb_func


def align_samples_func(ch, q_alignment,
                  samples,
                  aligned_samples,
                  original_samples,
                  Loc_Ref_Dict_All_Ch,
                  Loc_Morph_Ref_Dict_All_Ch,
                  funcs_dict, comp_func_dict,earth_models_dict,
                  gates_locations_dict,
                  n_sample,
                  verbose=False):

    '''

    :param ch:
    :param q_alignment:
    :param samples:
    :param aligned_samples:
    :param original_samples:
    :param Loc_Ref_Dict_All_Ch:
    :param Loc_Morph_Ref_Dict_All_Ch:
    :param funcs_dict:
    :param comp_func_dict:
    :param earth_models_dict:
    :param gates_locations_dict:
    :param n_sample:
    :param verbose:
    :return:
    '''

    Ref_Dict_indx_by_Loc_and_Morph = Loc_Morph_Ref_Dict_All_Ch[ch]
    types_of_alignment = []
    for targ_num in range(n_sample):
        if verbose:
            print(f'Sample {targ_num} **** \n')
        funcs = []
        z = samples[targ_num](ch)
        sample_orignal = copy.deepcopy(z)
        gates_locations = []
        xys = []
        refs = []

        for i in range(len(samples[targ_num].gates[ch])):
            target_gate = samples[targ_num].gates[ch][i]
            ind = np.logical_and(z>=target_gate.gate[0], z<target_gate.gate[1])
            target_seg1 = target_gate.segment
            target_ecdf = sm.distributions.empirical_distribution.ECDF(target_seg1)
            lo_gr, mr_gr = target_gate.location_group, target_gate.morphology_group
            ref_gate = Loc_Ref_Dict_All_Ch[ch][lo_gr]
            mr_gr_ref = Ref_Dict_indx_by_Loc_and_Morph.get((lo_gr, mr_gr))

            if verbose:
                print('diff:',np.abs(target_gate.segment-target_seg1).sum())
                print('lo_gr, mr_gr',lo_gr, mr_gr)
                print('\n mr_gr_ref', mr_gr_ref,'\n')
                print('target gate',target_gate,'\n')
            type_of_alignment = ''

            if mr_gr_ref:
                ref_seg1 = mr_gr_ref.segment
                if mr_gr_ref != target_gate:
                    ref_ecdf = sm.distributions.empirical_distribution.ECDF(ref_seg1)
                    ref_inv_cdf = sm.distributions.empirical_distribution.monotone_fn_inverter(ref_ecdf,
                                                                                               ref_seg1)
                    ref_inv_cdf = scipy.interpolate.BSpline(
                        ref_inv_cdf.x,
                        ref_inv_cdf.y,
                                                                     k=3,
                                                                     extrapolate=False
                                                                     )

                    func0 = lambda_warp_factory(ref_inv_cdf, target_ecdf)

                    type_of_alignment = f'Morph_IPT'
                else:
                    type_of_alignment = 'Mroph_Ref '
                    func0 = lambda x : x

            n_mi, n_mx = ref_gate.get_tight_gates(q=q_alignment)
            refs += [ref_gate.segment]
            if mr_gr_ref:
                funcf = minmaxmean_align(target_seg1 ,ref_gate.segment, n_mi, n_mx, q_alignment, func0)

                type_of_alignment = type_of_alignment + ' Loc_MinMax' if mr_gr_ref != target_gate else \
                    type_of_alignment + ' Loc_Ref'

            else:
                funcf = minmaxmean_align(target_seg1, ref_gate.segment, n_mi, n_mx,q_alignment)
                type_of_alignment ='minmax'

            funcs += [funcf]
            if verbose:
                print('loc. gate is alredy ref?', ref_gate == target_gate)
                print('gate:\n',target_gate.gate)
            sample_orignal[ind] = z[ind]
            t = np.linspace( target_gate.tight_gate[0], target_gate.tight_gate[1], 2000)
            y = funcf(t)
            ind = ~np.isnan(y)
            xys +=[[t[ind],y[ind]]]
            types_of_alignment +=[type_of_alignment]
            gates_locations += [[target_gate.gate[0], target_gate.gate[1]]]

        model = Earth(penalty=2, smooth=True, max_degree=1, )
        comp_func = combine_funcs(model, xys, funcs, gates_locations,plot=False)
        comp_func_dict[ch, targ_num] = comp_func
        earth_models_dict[ch, targ_num] = model
        funcs_dict[ch, targ_num] = funcs

        sample_aligned = comp_func(sample_orignal)
        aligned_samples[targ_num, ch] = sample_aligned
        original_samples[targ_num, ch] = sample_orignal
        gates_locations_dict[targ_num, ch] = gates_locations


