import numpy as np
import seaborn as sns
import skimage.segmentation as sk_seg
import copy
from skimage.filters import threshold_multiotsu
import jenkspy


class Gate:

    '''

        This is an abstraction class for a flow cytometry 1-D gate.

        :param segment: :obj:`numpy.array(dtype=float)`, shape = [n, 1], An array containing original data (cells) of a particular segment/gate.
        :param t: :obj:`numpy.array(dtype=float)`, shape = [m, 1], An array that represents the x axis. (pdf represents the y axis)
        :param pd: :obj:`numpy.array(dtype=float)`, shape = [m, 1], An array which represents probability density function.
        :param gate: obj:`list`: open-ended gate location.
        :param tight_gate: :obj:`list`: tight gate location.
        :param sample_num: :obj:`int`: the sample number of which this gate belongs to.
        :param ch_num: :obj:`int`: the channel number of which this gate belongs to.
        :param gate_num_in_sample: obj:`int`, gate number in sample.
        :param overall_indx_in_data: obj:`int`, the overall index of a gate in the dataset.
        :ivar seg_min: :obj:`float`, minimum value in segment.
        :ivar seg_max: :obj:`float`, maximum value in segment.
        :ivar location_group: :obj:`int`, the location group number this gate belongs to.
        :ivar group_ref_score: :obj:`float`, the gate score as a reference for its own location group.
        :ivar morphology_group: :obj:`int`, the morphology group number this gate belongs to.
        :ivar is_location_reference: :obj:`Bool `, if :obj:`True`, the gate is the reference for its location group.
        :ivar aligned_flag: :obj:`int `, flag indicating alignment status.
    '''

    def __init__(self,
                 segment,
                 t,pd,
                 gate,
                 tight_gate,
                 sample_num,
                 ch_num,
                 gate_num_in_sample,
                 overall_indx_in_data= None):


        self.segment = segment

        self.seg_min = segment.min()
        self.seg_max = segment.max()

        self.gate = gate
        self.tight_gate = tight_gate
        self.t, self.pd = t, pd
        self.sample_num = sample_num
        self.ch_num = ch_num
        self.gate_num_in_sample = gate_num_in_sample
        self.overall_indx = overall_indx_in_data
        self.location_group = 0
        self.group_ref_score = 0
        self.is_location_reference = False
        self.leave_to_MARS = False
        # morphology_group is relative to location_group
        self.morphology_group = 0
        self.aligned_flag = 0

    def get_tightened_segment(self, a=None, b= None):

        '''

        Get data of segment within the input bounds. Default behaviour returns segment within original tightened bounds.

        :param a: :obj:`float`, start location.
        :param b: :obj:`float`, end location.
        :returns:
         - bounded_segment - :obj:`numpy.array`, 1-D array of data within the specified bound.

        '''

        if a is None :
            a = self.tight_gate[0]
        if b is None:
            b = self.tight_gate[1]

        indx = np.logical_and(self.segment>=a, self.segment<b)
        bounded_segment = self.segment[indx]
        return bounded_segment

    def get_tight_gates(self, q=.01):

        '''
         Returns tight location of gate based on the provided quantile

        :param q: :obj:`float`, quantile value.
        :returns:
         - a - :obj:`float`, left boundary.
         - b - :obj:`float`, right boundary.

        '''

        a, b = np.quantile(self.segment, q), np.quantile(self.segment, 1-q)
        return a, b


    def __eq__(self, other):

        '''

        A dunder method that checks whether two gates are equal based on sample number, channel number and segment's start and end.

        :param other: :obj:`Gate`,
        :returns:
         -  - :obj:`Bool`, returns :obj:`True` if self is equal to other.

        '''

        if isinstance(other, Gate):
            return self.ch_num == other.ch_num and  \
                   self.sample_num == other.sample_num and self.gate == other.gate
        else:
            return False

    def __ne__(self, other):

        '''

        A dunder method that checks whether two gates are not equal if sample number, channel number nor segment's start and end are not equal.

        :param other: :obj:`Gate`,
        :returns:
         -  - :obj:`Bool`, returns :obj:`True` if self is not equal to other.
        '''

        return not self.__eq__(other)

    def __str__(self):

        '''

        returns string representation of object.

        :return: :obj:`String`
        '''

        s = f'(Sample_num: {self.sample_num}, Ch: {self.ch_num}, Gate_num: {self.gate_num_in_sample} \n' \
            f'Gate: [{int(1e3*self.gate[0])/1e3}, {int(1e3*self.gate[1])/1e3}], Segment_min_max: {int(1e3*self.seg_min)/1e3}, {int(1e3*self.seg_max)/1e3} \n'\
            f'Location_group: {self.location_group}, Morphology_group: {self.morphology_group}, ' \
            f'Overall_Indx: {self.overall_indx}, Aligned_flag: {self.aligned_flag})\n'
        return s
    def __repr__(self):
        return self.__str__()


class Sample:

    '''
        This is an abstraction class for a flow cytometry sample. It houses the raw data and also allows several levels of manipulation
        such as having access to particular channel's data, its probability density function and so forth.
        A sample is comprised of a number of channels. Each channel consists of a number of gates/segments.

        :param data: :obj:`numpy.array(dtype=float)`, shape = [m, ch], A 2-D array containing the original data (rows = cells, columns = channels). m = number of cells, ch= number of channels.
        :param gates_per_channel: :obj:`numpy.array(dtype=int)`, shape = [ch, 1], A 1-D array containing the number of gates in each channel. A '-1' entery means unspecified (Automatic Segmentation).
        :param sample_num: :obj:`int`, sample number in dataset.
        :param gate_factor_q: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing quantile value for each channel.
        :param kde_window: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing Kernel Density Estimate window size for each channel. This determines how coarse or granular the estimated pdf is.
        :param area_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed AUC (ratio of cells) to be considered a proper segment/gate. Only required for watershed segmentation.
        :param width_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed width of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param depth_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed depth of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel in sample.
        :ivar pdfs: :obj:`dict(list)`, a dictionary of the following pair [x, probability density functions] with channel number as its key.
        :ivar gates: :obj:`dict(list(Gate))`, a dictionary of lists of gates with channel number as its key.
        :ivar num_ch: :obj:`int`, number of channels.

    '''

    def __init__(self, data, gates_per_channel,sample_num, gate_factor_q= 0.04, area_threshold=.005,
                 kde_window= .3, width_threshold=.2, depth_threshold=.2, verbose= True):


        self.data = data
        self.sample_num = sample_num
        self.gates = {}
        self.all_bounds_t = {}
        self.tightened_gates = {}
        self.pdfs = {}
        num_ch = len(gates_per_channel)
        self.num_ch = num_ch
        self.gate_factor_q = np.ones(num_ch)*gate_factor_q if isinstance(gate_factor_q, float) else gate_factor_q
        self.area_threshold = np.ones(num_ch)*area_threshold if isinstance(area_threshold, float) else area_threshold
        self.kde_window = np.ones(num_ch)*kde_window if isinstance(kde_window, float) else kde_window
        self.width_threshold = np.ones(num_ch)*width_threshold if isinstance(width_threshold, float) else width_threshold
        self.depth_threshold = np.ones(num_ch)*depth_threshold if isinstance(depth_threshold, float) else depth_threshold

        for ch, n in enumerate(gates_per_channel):
            self.resegment_ch(ch, init=True, verbose=verbose)

           # z_t = data[:,i]
           # print('ch:',i)
           #
           # kde = sns._statistics.KDE(bw_method=self.kde_window[i])
           # pd, t = kde(x1=z_t.flatten())
           # self.pdfs[i] = [t, pd]
           #
           # if n!=-1:
           #     if n>1:
           #         # bounds_t = segment_func(z_t, method='otsu', num_segments=n)
           #         bounds_t,t,pd = segment_func(z_t, method='je', num_segments=n)
           #     else:
           #         bounds_t,t,pd = [z_t.min(), z_t.max()]
           # else:
           #
           #     bounds_t = segment_func(z_t, method='water', kde_window= self.kde_window[i],
           #                        area_threshold=self.area_threshold[i],
           #                        width_threshold=self.width_threshold[i],
           #                        depth_threshold =self.depth_threshold[i])
           #
           # bounds_t = convert_watershed_lines_to_gate(bounds_t)
           # self.all_bounds_t[i] = bounds_t
           # print('bounds', bounds_t)
           # tightened_bounds = tighten_all_gates(z_t.reshape(-1,1), bounds_t, gate_factor_q[i])
           # print('tightened_bounds', tightened_bounds)
           # for k, b in enumerate(bounds_t):
           #      ind = np.logical_and(z_t>=b[0],
           #                                   z_t<b[1])
           #      seg = z_t[ind]
           #      gate_obj  = Gate(seg, t,pd,
           #                       gate=b,
           #                       tight_gate=tightened_bounds[k],
           #                             sample_num=sample_num,
           #                       ch_num=i,
           #                       gate_num_in_sample=k)
           #      if self.gates.get(i):
           #         self.gates[i] += [gate_obj]
           #      else:
           #         self.gates[i] =  [gate_obj]


    def resegment_ch(self, ch, kde_window=None, area_threshold=None, width_threshold=None,
                        depth_threshold=None, init=False, verbose=True):

        '''

        Resegments channel based on new threshold values.

        :param ch: :obj:`int`, channel number.
        :param kde_window: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing Kernel Density Estimate window size for each channel. This determines how coarse or granular the estimated pdf is.
        :param area_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed AUC (ratio of cells) to be considered a proper segment/gate. Only required for watershed segmentation.
        :param width_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed width of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param depth_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed depth of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param init: :obj:`Bool`, if :obj:`True`, recomputes probability density function.
        :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel in sample.
        '''

        if not init and kde_window is None and area_threshold and None and width_threshold is None and depth_threshold is None :
            return

        self.area_threshold[ch] = self.area_threshold[ch] if area_threshold is None else area_threshold
        self.width_threshold[ch] = self.width_threshold[ch] if width_threshold is None else width_threshold
        self.depth_threshold[ch] = self.depth_threshold[ch] if depth_threshold is None else depth_threshold

        z_t = self.data[:, ch]
        if kde_window is None and not init:
            x, pdf = self.pdfs[ch]
        elif init or kde_window is not None:
            self.kde_window[ch] = self.kde_window[ch] if kde_window is None else kde_window

            kde = sns._statistics.KDE(bw_method=self.kde_window[ch])
            pdf, x = kde(x1=z_t.flatten())
            self.pdfs[ch] = [x, pdf]

        bounds_t = segment_func(z_t, method='water',
                                x=x, pdf=pdf,
                                area_threshold=self.area_threshold[ch],
                                width_threshold=self.width_threshold[ch],
                                depth_threshold=self.depth_threshold[ch],
                                verbose = verbose
                                )
        bounds_t = convert_watershed_lines_to_gate(bounds_t)
        self.all_bounds_t[ch] = bounds_t
        tightened_bounds = tighten_all_gates(z_t.reshape(-1, 1), bounds_t, self.gate_factor_q[ch])
        if verbose:
            print(self.area_threshold[ch], self.width_threshold[ch], self.depth_threshold[ch])
            print('bounds', bounds_t)
            print('tightened_bounds', tightened_bounds)
        self.gates[ch] = []
        for k, b in enumerate(bounds_t):
            ind = np.logical_and(z_t >= b[0],
                                 z_t < b[1])
            seg = z_t[ind]
            gate_obj = Gate(seg, x, pdf,
                            gate=b,
                            tight_gate=tightened_bounds[k],
                            sample_num=self.sample_num,
                            ch_num=ch,
                            gate_num_in_sample=k)
            self.gates[ch] += [gate_obj]


    def tighten_gates_in_a_channel(self, ch, gate_factor_q):

            '''
                Updates the tight gate location for gates of a particular channels based on the provided quantile.

                :param ch: :obj:`int`, channel number.
                :param gate_factor_q: :obj:`float`, quantile value.
            '''

            z_t = self.data[:,ch]
            tightened_bounds = tighten_all_gates(z_t, self.all_bounds_t[ch], gate_factor_q)
            for k, b in enumerate(tightened_bounds):
               self.gates[ch][k].tight_gate = b


    def tighten_all_gates(self, gate_factor_q):

        '''
            Updates the tight gate location for a all channels based on the provided quantile.

            :param gate_factor_q: :obj:`float`, quantile value.
        '''

        for ch in range(self.ch_num):
            self.tighten_gates_in_a_channel(ch, gate_factor_q)

    def update_gate(self, channel, index, flag=1):

        '''

          Updates the status of only the gate according to channel number and index with the provided flag.

          :param channel: :obj:`int`, channel number.
          :param index: :obj:`int`, gate index.
          :param flag: :obj:`int`, alignment flag.
        '''

        self.gates[channel][index].aligned_flag = flag

    def update_channel_gates(self, channel, flag=1):

        '''

          Updates the status of all gates in the provided channel with the provided flag.

          :param channel: :obj:`int`, channel number.
          :param flag: :obj:`int`, alignment flag.
        '''
        for gate in self.gates[channel]:
            gate.aligned_flag = flag




    def check_if_still_non_aligned(self, channel):

        '''

        Checks whether all gates in the provided channel have been aligned or not.

        :param channel: :obj:`int`, channel number.
        :return: :obj:`Bool`.
        '''

        b = False
        for g in self.gates[channel]:
            b = b | ~ g.aligned_flag
        return b

    def get_ch_alignment_status(self, channel):

        '''

        Check whether all gates in channel have been aligned or not.

        :param channel: :obj:`int`, channel number.
        :return: :obj:`Bool`.
        '''

        b = 0
        for g in self.gates[channel]:
            b+=g.aligned_flag
        return b/len(self.gates[channel])


    # def revert_min_max_normalization(self, maxes, mins):
    #     for ch, _ in enumerate(maxes):
    #         self.data[:,ch] = self.data[:,ch]*(maxes[ch]-mins[ch]) + mins[ch]
    #
    #     for ch, value in self.gates.items():
    #         for gate in value:
    #             a,b = (maxes[ch]-mins[ch]),  mins[ch]
    #             gate[1] = [ g*a + b for g in gate[1]]
    #     return self

    def __call__(self, ch):
        return self.data[:,ch]


def non_maximum_suppression(pdf, seg_pdf, area_threshold = 0.01):

    '''

    Performs non-maximum suppression on segments that do not surpass the provided area-under-the-curve (AUC) threshold.

    :param pdf: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which represents probability density function.
    :param seg_pdf: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which contains segment labels for each point in the provided pdf.
    :param area_threshold: :obj:`float`: the minimum allowed AUC (ratio of cells) to be considered a proper segment/gate.
    :return: :obj:`numpy.array(dtype=int)`, shape = [m, 1], masked segments with zeros in place of the original suppressed segments.
    '''

    seg_mask = copy.deepcopy(seg_pdf)
    for s in np.unique(seg_pdf):
        # compute the normalized AUC using trapezoid method.
        r = np.trapz(pdf[seg_mask==s])/np.trapz(pdf)
        # suppress if r is less than threshold or the segment has less than 2 elements.
        if r <=area_threshold or np.count_nonzero(seg_mask==s)<2:
            seg_mask[seg_mask==s] = 0
    return seg_mask


def merge(seg_pdf):

    '''

    Assign each point in segments of seg_pdf that has been suppressed (i.e. zeroed)  to the closest segment.

    :param seg_pdf: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which contains segment labels for each point in pdf.
    :return: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which contains newly assigned segment labels.
    '''


    d = np.diff(seg_pdf)
    ind = np.argwhere(d).flatten()+1
    ind = np.concatenate([[0],ind, [len(seg_pdf)]])
    arr = copy.deepcopy(seg_pdf)
    for i in range(len(ind)-1):
        if seg_pdf[ind[i]]==0:
            if i ==0:
                b = seg_pdf[ind[i+1]+1]
                arr[:ind[i+1]] = b
            elif i==len(ind)-2:
                b = seg_pdf[ind[i]-1]
                arr[ind[i]:] = b
            else:
                a, b = seg_pdf[ind[i]-1], seg_pdf[ind[i+1]+1]
                a_ind, b_ind= ind[i], ind[i+1]
                m_ind = a_ind + (b_ind-a_ind)//2
                arr[a_ind:m_ind] = a
                arr[m_ind: b_ind] = b
    return arr


def suppress_and_merge_tiny_basins(pdf, seg_pdf, width_threshold, depth_threshold, verbose=True):

    '''

    Suppresses and merges tiny basins/gates/segments depending on the width and depth of the ravine between them.

    :param pdf: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which represents probability density function.
    :param seg_pdf: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which contains segment labels for each point in the provided pdf.
    :param width_threshold: :obj:`float`: the minimum allowed width of the ravine between two segments/gates for the later to be considered a separate  segment or an extension of the previous segment.
    :param depth_threshold: :obj:`float`: the minimum allowed depth of the ravine between two segments/gates for the later to be considered a separate  segment or an extension of the previous segment.
    :param verbose: :obj:`Bool`, if :obj:`true`, prints width and depth values and whether they are suppressed or not.
    :return: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which contains newly assigned segment labels.
    '''

    # There may be a bug in this method that needs fixing.
    # a possible improvement is to merge segments using agglomerative clustering

    new_seg_mask = copy.deepcopy(seg_pdf)
    d = np.diff(seg_pdf)
    watershed_lines = np.argwhere(d).flatten()
    seg_labels = np.unique(seg_pdf)
    merge_map = [0]*(len(seg_labels)-1)
    for i in range(1,len(seg_labels)):
        curr_seg, prev_seg= seg_labels[i], seg_labels[i-1]
        curr_ind, prev_ind = seg_pdf==curr_seg, seg_pdf==prev_seg
        curr_max, prev_max = np.argmax(pdf[curr_ind])+ np.argmax(curr_ind),\
                             np.argmax(pdf[prev_ind]) + np.argmax(prev_ind)
        dist_cond = np.abs(curr_max - prev_max)/len(pdf) < width_threshold
        depth = np.abs(np.maximum(pdf[curr_max], pdf[prev_max]) - pdf[watershed_lines[i-1]])/pdf.max()
        if verbose:
            print('width', np.abs(curr_max - prev_max)/len(pdf), 'depth', depth,
                  'width_cond', dist_cond)
        if depth < depth_threshold or dist_cond:
            merge_map[i-1]=1
    start_seg = seg_labels[0]
    for i, flag in enumerate(merge_map):
        if flag:
            s = seg_labels[i+1]
            new_seg_mask[new_seg_mask==s] = start_seg
        else:
            start_seg = i+1

    return new_seg_mask


def get_watershed_lines_from_labeled_segs(seg_pdf, x):

    '''

    Gets watershed lines locations from segmented probability density function.

    :param seg_pdf: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which contains segment labels for each point in the provided pdf.
    :param x: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array that represents the x axis. (seg_pdf represents the y axis)
    :return gates: :obj:`list`, a list with its entries as the boundaries of each gate
    '''

    d = np.diff(seg_pdf)
    ind = np.argwhere(d).flatten()+1
    ind = np.concatenate([[0], ind, [len(seg_pdf)-1]])
    watershed_lines = [x[g] for g in ind]
    return watershed_lines


def tighten_all_gates(z, gates, gate_q_factor = .01):

    '''

    Tightens all gates location based on the provided quantile.

    :param z: :obj:`numpy.array(dtype=float)`, shape = [n, 1], An array which contains original data (cells).
    :param gates: :obj:`list`, :obj:`len(gates) = m` where m is the number of gates.
    :param gate_q_factor: :obj:`float`, tightening the gates using quantiles.
    :return: modified_gates :obj:`list`, :obj:`len(gates) = m` where m is the number of gates.
    '''

    modified_gates = copy.deepcopy(gates)
    for i, gate in enumerate(gates):
        ind = np.logical_and( z>=gate[0], z<=gate[1])
        q0 = np.quantile(z[ind], gate_q_factor)
        q1 = np.quantile(z[ind], 1-gate_q_factor)
        gate_a = copy.deepcopy(gate)
        gate_a[0] = q0
        gate_a[1] = q1
        modified_gates[i] = gate_a
    return modified_gates


def convert_watershed_lines_to_gate(watershed_lines):

    '''

    Given watershed lines such as [0, 0.5, 1], this method returns the following list of gates [[0,.5], [.5,1]].

    :param watershed_lines: :obj:`list`, :obj:`len(bounds) = m` where m is the number of watershed lines.
    :return gates: :obj:`list(list)`, :obj:`len(gates) = m-1` where m is the number of gates.
    '''

    gates = []
    for i,b in enumerate(watershed_lines[:-1]):
        g = [watershed_lines[i], watershed_lines[i+1]]
        gates += [g]
    return gates


def convert_gates_to_watershed_lines(gates):

    '''

    Given a list of gates such as [[0,.5], [.5,1]] this method returns the following list of watershed lines [0, 0.5, 1].

    :param gates: :obj:`list(list)`, :obj:`len(gates) = m` where m is the number of gates.
    :return watershed_lines: :obj:`list`, :obj:`len(watershed_lines) = m+1` where m is the number of watershed lines.
    '''

    watershed_lines = []
    for i in range(0, len(gates)):
        g = [gates[i][0]]
        watershed_lines += g
    watershed_lines += [gates[-1][1]]
    return watershed_lines


def segment_func(z, method = 'water',
            x=None, pdf=None,
            area_threshold=.005,
            width_threshold=.2,
            depth_threshold=.2,
            num_segments = 2,
            verbose=True
            ):

    '''
    This method performs segmentation on a raw vector of cells.

    :param z: :obj:`numpy.array(dtype=float)`, shape = [n, 1], An array which contains original data (cells).
    :param method: :obj:`str`: defines segmentation method, ['watershed', 'otsu', 'jenkis'].
    :param x: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array that represents the x axis where pdf represents the y axis. (Only required for watershed segmentation.)
    :param pdf: :obj:`numpy.array(dtype=int)`, shape = [m, 1], An array which represents the probability density function. (Only required for watershed segmentation.)
    :param area_threshold: :obj:`float`: the minimum allowed AUC (ratio of cells) to be considered a proper segment/gate. (Only required for watershed segmentation.)
    :param width_threshold: :obj:`float`: the minimum allowed width of the ravine between two segments/gates for the later to be considered a separate  segment or an extension of the previous segment. Only required for watershed segmentation.
    :param depth_threshold: :obj:`float`: the minimum allowed depth of the ravine between two segments/gates for the later to be considered a separate  segment or an extension of the previous segment. Only required for watershed segmentation.
    :param num_segments: :obj:`int`: number of segments. Only required for Otsu and Jenkis.
    :param verbose: :obj:`Bool`, if :obj:`true`, prints each gate area, width and depth values and whether they are suppressed or not.
    :returns:
        - watershed_lines - :obj:`list`, :obj:`len(bounds) = m+1`.
    '''

    method = method.lower()


    if method == 'otsu':
        watershed_lines = threshold_multiotsu(z, int(num_segments), 128)
        watershed_lines = np.concatenate([[z.min()],watershed_lines,[z.max()]])

    elif method == 'je' or method == 'jenkis':
        # subsample
        ind = np.random.randint(0, z.shape[0], np.minimum(z.shape[0], 10000))
        watershed_lines = jenkspy.jenks_breaks(z[ind],int(num_segments))

    elif method == 'water' or method == 'watershed':
        # flip and segment
        pd_a_s = sk_seg.watershed(-pdf)
        # apply non-maximum suppression on area
        pd_a_s =  non_maximum_suppression(pdf, pd_a_s, area_threshold = area_threshold)
        # merge suppressed segments
        pd_a_s = merge(pd_a_s)
        # suppress and merge segments with tiny ravines between them.
        pd_a_s = suppress_and_merge_tiny_basins(pdf, pd_a_s, width_threshold, depth_threshold,verbose)
        # make segments starts from 0.
        pd_a_s -=pd_a_s.min()
        watershed_lines = get_watershed_lines_from_labeled_segs(pd_a_s, x)
    return watershed_lines
