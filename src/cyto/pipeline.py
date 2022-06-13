from sklearn.preprocessing import MinMaxScaler
import segmentation
from segmentation import *
from grouping import *
from alignment import *
from importlib import reload
reload(segmentation)

import time

class Pipeline:

    '''

        This is a class that represents a complete alignment pipeline. To align a dataset, create an instance of this class with your data and hyperparameters.
        There are two modes of using the pipeline class, end-to-end and manual. For end-to-end, create an instance of the pipeline class with segment_data = True and run end_to_end_align after.
        The manual mode gives the user more granular control to make sure that every satge of the pipeline is producing optimal results.


        :param X_data: :obj:`numpy.array(dtype=float)`, shape = [ch, n], A 2-D array containing the original, raw, unormalized data (rows = cells, columns = channels). n = number of cells for all samples, ch= number of channels.
        :param Y: :obj:`numpy.array(dtype=int)`, shape = [1, n], A 1-D array containing sample numbers.
        :param num_channels: :obj:`int`, number of channels.
        :param feature_range: :obj:`tuple`, a tuple indicating minimum and maximum range for minmax normalization, default = (0,1).
        :param channel_names: :obj:`list(str)`, :obj:`len(channel_names) = ch`, a list of strings indicating each channel name.
        :param gate_factor_q: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing quantile value for each channel.
        :param kde_window: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing Kernel Density Estimate window size for each channel. This determines how coarse or granular the estimated pdf is.
        :param area_thresholds: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed AUC (ratio of cells) to be considered a proper segment/gate. Only required for watershed segmentation.
        :param width_thresholds: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed width of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param depth_thresholds: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed depth of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param jaccard_thresholds: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing Jaccard thresholds for each channel.
        :param wass_dist_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing wasserstien distance thresholds for each channel.
        :param sigma: :obj:`float`, the standard deviation of Gaussian kernel.
        :param earth_smoothing_penalty: :obj:`float`, A smoothing parameter used to calculate generalized cross validation. Used during the pruning pass and to determine whether to add a hinge or linear basis function during the forward pass.
        :param segment_data: :obj:`Bool`, segment data after creating the pipeline object. Set as :obj:`True` for end-to-end alignment.
        :ivar min_before_norm: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum value of each channel before minmax.
        :ivar max_before_norm: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the maximum value of each channel before minmax.
        :ivar num_samples: :obj:`int`, number of samples.
        :ivar samples: :obj:`list(Sample)`, a list of all samples in data as :obj:`Sample` objects.
        :ivar Sim_Matrix_dict: :obj:`dict`, a dictionary of  [n,n] :obj:`numpy.array` Jaccard Similarity Matrices between pairs of gates with channels numbers as its keys where n is the number of gates.
        :ivar agg_models_dict: :obj:`dict`, a dictionary of computed :obj:`sklearn.cluster.AgglomerativeClustering` Agglomerative Models based on the computed Jaccard Similarity Matrices with channels numbers as its keys.
        :ivar location_groups_dict: :obj:`dict`, a dictionary of :obj:`int` location groups for each channel with channels numbers as its keys.
        :ivar Loc_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of :obj:`Gate` chosen reference gates for all channels with channels numbers as its keys.
        :ivar incidence_matrices: :obj:`dict`, a dictionary of  [m,m] :obj:`numpy.array` incidence matrices with channels numbers as its keys where m is the number of groups.
        :ivar loosened_groups_for_deadlock_dict: :obj:`dict`, a dictionary of constraints that have been loosened because of a deadlock with channels numbers as its keys.
        :ivar Morph_models_dict: :obj:`dict`, a dictionary of computed :obj:`sklearn.cluster.AgglomerativeClustering` Agglomerative Models based on the computed Wasserstein Distance Matrices with channels numbers as its keys.
        :ivar Loc_Morph_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of :obj:`Gate` chosen morphology reference gates for all channels with (channel number, location group) tupel as its keys.
        :ivar Morph_groups_All_ch: :obj:`dict`, a dictionary of :obj:`int` morphology groups for each channel with channels numbers as its keys.
        :ivar aligned_samples: :obj:`dict`, a dictionary of :obj:`numpy.array(dtype=float)` aligned_samples,  with its keys is represented as the tupel (sample number, channel).
        :ivar original_samples: :obj:`dict`, a dictionary of :obj:`numpy.array(dtype=float)` original_samples, morphology groups for each channel with channels numbers as its keys.
        :ivar gates_locations_dict: :obj:`dict`, a dictionary of :obj:`list` location of gates,  with its keys is represented as the tupel (sample number, channel).
        :ivar funcs_dict: :obj:`dict`, a dictionary of :obj:`list` warping functions for each gate in a channel of a sample, with its keys is represented as the tupel (sample number, channel).
        :ivar comp_func_dict: :obj:`dict`, a dictionary of :obj:`function` combined warping functions for each channel in each sample, with its keys is represented as the tupel (sample number, channel).
        :ivar earth_models_dict: :obj:`dict`, a dictionary of :obj:`pyearth.Earth` MARS models for each channel in each sample, with its keys is represented as the tupel (sample number, channel).

    '''

    def __init__(self, X_data, Y, num_channels, feature_range =(0,1), channel_names= None,
                            gate_factor_q=.02, area_thresholds=.04,
                            kde_window=.28, width_thresholds=.16,
                            depth_thresholds=.23, jaccard_thresholds=.6, wass_dist_threshold = 1e-9,
                 sigma=1, earth_smoothing_penalty=2, segment_data= False

                 ):
        # preprocessing and organizing

        self.max_before_norm, self.min_before_norm = X_data.max(0), X_data.min(0)
        self.feature_range = feature_range
        self.X = MinMaxScaler(feature_range=feature_range).fit_transform(X_data)
        self.Y = Y

        # meta-data containers

        self.num_channels = num_channels
        self.num_samples = len(np.unique(Y))
        self.channel_names = channel_names if channel_names is not None else list(range(num_channels))

        # Samples container

        self.samples = []

        # Segmentation hyperparameters

        self.gate_factor_q = np.ones(self.num_channels) * gate_factor_q if isinstance(gate_factor_q, float) else gate_factor_q
        self.kde_windows = np.ones(self.num_channels) * kde_window if isinstance(kde_window, float) else kde_window
        self.area_thresholds = np.ones(self.num_channels) * area_thresholds if isinstance(area_thresholds, float) else area_thresholds
        self.width_thresholds = np.ones(self.num_channels) * width_thresholds if isinstance(width_thresholds,
                                                                              float) else width_thresholds
        self.depth_thresholds = np.ones(self.num_channels) * depth_thresholds if isinstance(depth_thresholds, float) else depth_thresholds

        # containers for location grouping and ranking

        self.Sim_Matrix_dict = {}
        self.agg_models_dict = {}
        self.location_groups_dict = {}
        self.Loc_Ref_Dict_All_Ch = {}
        self.incidence_matrices = {}
        self.loosened_groups_for_deadlock_dict = {}
        self.jaccard_thresholds = [jaccard_thresholds]*num_channels

        # containers for morphology

        self.Morph_models_dict = {}
        self.Loc_Morph_Ref_Dict_All_Ch = {}
        self.Morph_groups_All_ch = {}

        self.wass_dist_threshold = np.ones(num_channels) * wass_dist_threshold

        # containers for alignment parameters

        self.aligned_samples = {}
        self.original_samples = {}
        self.gates_locations_dict = {}
        self.funcs_dict = {}
        self.comp_func_dict = {}
        self.Ref_Inv_CDF_Dict_All_Ch = {}
        self.earth_models_dict = {}
        self.sigmas = np.ones(self.num_channels) * sigma if isinstance(sigma, float) else sigma
        self.earth_smoothing_penalty = np.ones(self.num_channels) * earth_smoothing_penalty if isinstance(earth_smoothing_penalty, float) else earth_smoothing_penalty

        if segment_data:
            self.segment_data(gate_factor_q = self.gate_factor_q, area_thresholds=self.area_thresholds, kde_window= self.kde_windows,
                              width_thresholds=self.width_thresholds, depth_thresholds=self.depth_thresholds)


    def segment_data(self, gate_factor_q=.02, area_thresholds=.04,
                            kde_window=.28, width_thresholds=.16,
                            depth_thresholds=.23, commit_changes= True, verbose= True):

        '''

        Initial segmentation. Creates Sample and Gate objects to be modified later.

        :param gate_factor_q: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing quantile value for each channel.
        :param area_thresholds: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed AUC (ratio of cells) to be considered a proper segment/gate. Only required for watershed segmentation.
        :param kde_window: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing Kernel Density Estimate window size for each channel. This determines how coarse or granular the estimated pdf is.
        :param width_thresholds: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed width of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param depth_thresholds: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing the minimum allowed depth of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param commit_changes: :obj:`Bool`, if :obj:`True`, commits changes to :obj:`Datahandler` object.
        :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel for every sample.
        '''

        self.gate_factor_q = np.ones(self.num_channels) * gate_factor_q if isinstance(gate_factor_q, float) else gate_factor_q
        self.kde_windows = np.ones(self.num_channels) * kde_window if isinstance(kde_window, float) else kde_window
        self.area_thresholds = np.ones(self.num_channels) * area_thresholds if isinstance(area_thresholds, float) else area_thresholds
        self.width_thresholds = np.ones(self.num_channels) * width_thresholds if isinstance(width_thresholds,
                                                                              float) else width_thresholds
        self.depth_thresholds = np.ones(self.num_channels) * depth_thresholds if isinstance(depth_thresholds, float) else depth_thresholds

        time_start = time.time()

        for y in range(self.num_samples):
            s_d = self.X[self.Y == y, :]
            print('****\n' * 4, 'Sample:', y, '****\n' * 4)
            sample = Sample(s_d, gates_per_channel=[-1] * self.num_channels, sample_num=y,
                            gate_factor_q=self.gate_factor_q, area_threshold=self.area_thresholds,
                            kde_window=self.kde_windows, width_threshold=self.width_thresholds,
                            depth_threshold=self.depth_thresholds)
            self.samples += [sample]

        time_end = time.time()
        print(time_end-time_start)

        if commit_changes:
            self.commit_changes(verbose)


    def resegment_channel(self, channel, area_threshold=None, width_threshold =None,
                                depth_threshold=None, kde_window=None, verbose= False, init=False):

        '''

        Resegments channel based on new threshold values.

        :param channel: :obj:`int`, channel number.
        :param area_thresholds: :obj:`float`, shape = [ch, 1], A 1-D array containing the minimum allowed AUC (ratio of cells) to be considered a proper segment/gate. Only required for watershed segmentation.
        :param kde_window: :obj:`float`, shape = [ch, 1], A 1-D array containing Kernel Density Estimate window size for each channel. This determines how coarse or granular the estimated pdf is.
        :param width_thresholds: :obj:`float`, shape = [ch, 1], A 1-D array containing the minimum allowed width of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param depth_thresholds: :obj:`float`, shape = [ch, 1], A 1-D array containing the minimum allowed depth of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel for every sample.
        :param init: :obj:`Bool`, if :obj:`True`, recomputes pdf.
        '''

        self.area_thresholds[channel]  = self.area_thresholds[channel]  if area_threshold  is None else area_threshold
        self.width_thresholds[channel] = self.width_thresholds[channel] if width_threshold is None else width_threshold
        self.depth_thresholds[channel] = self.depth_thresholds[channel] if depth_threshold is None else depth_threshold

        if kde_window is not None:
            self.kde_windows[channel] = kde_window

        for i, s in enumerate(self.samples):
            s.resegment_ch(channel, kde_window=kde_window, area_threshold=area_threshold,
                           width_threshold=width_threshold,
                           depth_threshold=depth_threshold,
                           init=init, verbose= verbose)

    def compute_location_morphology_groups_from_manual_refs(self, channels, ref_samples, jaccard_thresholds, wass_dist_thresholds):

        '''

        :param channels: :obj:`list(int)`, a list of channels to update.
        :param ref_samples: Union(:obj:`int`, :obj:`list`), number/s of reference sample/s for each channel.
        :param jaccard_thresholds: :obj:`dict`, a dictionary of Jaccard thresholds indicating where to cut the dendrogram with channels numbers as its keys.
        :param wass_dist_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing wasserstien distance thresholds for each channel.
        '''

        set_references_manually(channels, ref_samples,
                                data_handler=self.data_handler, jaccard_thresholds=jaccard_thresholds,
                                wass_dist_thresholds=wass_dist_thresholds,
                                Loc_Ref_Dict_All_Ch = self.Loc_Ref_Dict_All_Ch,
                                Loc_Morph_Ref_Dict_All_Ch= self.Loc_Morph_Ref_Dict_All_Ch, wass_n_samples=2000
                                )

    def recompute_and_update_location_hierarchy_and_refs(self, channels, jaccard_thresholds):

        '''

         Recomputes references for the provided channels and updates Similarity Matrix Dictionary, Agglomerative Model Dictionary, Location Groups Dictionary,
         Location References Dictionary, Incidence Matrcies Dictionary, Loosened Groups For Deadlock Dictionary.

        :param channels: :obj:`list(int)`, a list of channels to update.
        :param jaccard_thresholds: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing Jaccard thresholds indicating where to cut the dendrogram with channels numbers as its keys.

        '''

        for i, ch in enumerate(channels):
            self.jaccard_thresholds[ch] = jaccard_thresholds if isinstance(jaccard_thresholds,
                                                                              float) else jaccard_thresholds[i]
        recompute_and_update_location_hierarchy_and_refs(
            channels=channels,
            data_handler = self.data_handler,
            jaccard_thresholds = self.jaccard_thresholds,
            Sim_Matrix_dict = self.Sim_Matrix_dict,
            agg_models_dict = self.agg_models_dict,
            location_groups_dict = self.location_groups_dict,
            Loc_Ref_Dict_All_Ch = self.Loc_Ref_Dict_All_Ch,
            loosened_groups_for_deadlock_dict = self.loosened_groups_for_deadlock_dict,
            incidence_matrices = self.incidence_matrices
        )

    def update_morphology_hierarchy_and_refs(self, channels, wass_dist_threshold=1e-9):

        '''

        Recomputes morphology references for the provided channels and updates Agglomerative Model Dictionary, Morphology Group Dictionary,
        and Morphology References Dictionary.


        :param channels: :obj:`list(int)`, a list of channels to update.
        :param wass_dist_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing wasserstien distance thresholds for each channel.
        '''

        self.wass_dist_threshold = np.ones(self.num_channels) * wass_dist_threshold if isinstance(wass_dist_threshold,
                                                                              float) else wass_dist_threshold

        update_morphology_hierarchy_and_refs(channels= channels,
                                             wass_dist_threshold= self.wass_dist_threshold,
                                             data_handler=self.data_handler,
                                             location_groups_dict=self.location_groups_dict,
                                             Loc_Morph_Ref_Dict_All_Ch=self.Loc_Morph_Ref_Dict_All_Ch,
                                             Morph_models_dict=self.Morph_models_dict,
                                             Morph_groups_All_ch=self.Morph_groups_All_ch
                                             )

    def align_samples(self, channels, sigma =1,  earth_smoothing_penalty=2, n_sample =-1, subsample_ratio=1, verbose= False):

        '''

        Transforms channel ch unaligned data to aligned data based on the provided references and fills in the provided aligned samples dictionary.

        :param channels: :obj:`list(int)`, a list of channels to update.
        :param sigma: :obj:`float`, the standard deviation of Gaussian kernel.
        :param earth_smoothing_penalty: :obj:`float`, A smoothing parameter used to calculate generalized cross validation. Used during the pruning pass and to determine whether to add a hinge or linear basis function during the forward pass.
        :param n_sample: :obj:`int`, number of samples to align, (default = -1), align all samples.
        :param subsample_ratio: :obj:`float`, a number between [0,1]`, represents the ratio of cells to consider to estimate MARS parameters.
        :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel for every sample.
        '''

        n_sample = self.num_samples if n_sample ==-1 else n_sample
        for ch in channels:
            print(channels, ch)
            align_samples_func(ch=ch, q_alignment = self.gate_factor_q[ch],
                               samples=self.samples, aligned_samples=self.aligned_samples,
                               original_samples=self.original_samples,
                               Loc_Ref_Dict_All_Ch=self.Loc_Ref_Dict_All_Ch,Ref_Inv_CDF_Dict_All_Ch=self.Ref_Inv_CDF_Dict_All_Ch,
                               Loc_Morph_Ref_Dict_All_Ch=self.Loc_Morph_Ref_Dict_All_Ch,
                               funcs_dict=self.funcs_dict, comp_func_dict=self.comp_func_dict,
                               gates_locations_dict=self.gates_locations_dict,
                               earth_models_dict=self.earth_models_dict,
                               n_sample = n_sample, sigma=sigma,
                               earth_smoothing_penalty=earth_smoothing_penalty,
                               subsample_ratio = subsample_ratio,
                               verbose = verbose)

        self.commit_changes(verbose)

    def check_create_arr_for_hyperparam(self, hyperparam):

        '''

        Checks whether the hyperparameter is None, int, float, or array and returns an iterable of the appropriate size.

        :param hyperparam: Union(:obj:`None`, :obj:`int`, :obj:`float`, :obj:`list`, :obj:`numpy.array`)
        '''

        if hyperparam is None:
            return [None] * self.num_channels
        elif isinstance(hyperparam, (int, float)):
            return hyperparam * np.ones(shape=self.num_channels)
        else:
            return hyperparam

    def end_to_end_align(self, channels, area_thresholds=None, width_thresholds =None,
                                depth_thresholds=None, kde_windows=None, jaccard_thresholds =None, wass_dist_thresholds = None, ref_samples_manual=None,
                         sigma=1, earth_smoothing_penalty=2, n_sample=-1, subsample_ratio=1, verbose = True
                         ):


        '''

         Performs segmentation, choosing references and transforming the data automatically using the provided hyperparamters.

        :param channels: :obj:`list(int)`, a list of channels to update.
        :param area_thresholds: :obj:`float`, shape = [ch, 1], A 1-D array containing the minimum allowed AUC (ratio of cells) to be considered a proper segment/gate. Only required for watershed segmentation.
        :param width_thresholds: :obj:`float`, shape = [ch, 1], A 1-D array containing the minimum allowed width of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param depth_thresholds: :obj:`float`, shape = [ch, 1], A 1-D array containing the minimum allowed depth of the ravine between two segments/gates for the later to be considered a separate segment or an extension of the previous segment. Only required for watershed segmentation.
        :param kde_window: :obj:`float`, shape = [ch, 1], A 1-D array containing Kernel Density Estimate window size for each channel. This determines how coarse or granular the estimated pdf is.
        :param jaccard_thresholds: :obj:`dict`, a dictionary of Jaccard thresholds indicating where to cut the dendrogram with channels numbers as its keys.
        :param wass_dist_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing wasserstien distance thresholds for each channel.
        :param ref_samples_manual: :obj:`int` or :obj:`list(int)`, a list of manual references numbers for each channel. (default = :obj:`None`) implies find references automatically.
        :param sigma: :obj:`float`, the standard deviation of Gaussian kernel.
        :param earth_smoothing_penalty: :obj:`float`, A smoothing parameter used to calculate generalized cross validation. Used during the pruning pass and to determine whether to add a hinge or linear basis function during the forward pass.
        :param n_sample: :obj:`int`, number of samples to align, (default = -1), align all samples.
        :param subsample_ratio: :obj:`float`, a number between [0,1]`, represents the ratio of cells to consider to estimate MARS parameters.
        :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel for every sample.
        '''

        area_thresholds = self.check_create_arr_for_hyperparam(area_thresholds)
        width_thresholds = self.check_create_arr_for_hyperparam(width_thresholds)
        depth_thresholds = self.check_create_arr_for_hyperparam(depth_thresholds)
        kde_windows = self.check_create_arr_for_hyperparam(kde_windows)
        jaccard_thresholds = self.check_create_arr_for_hyperparam(jaccard_thresholds)
        wass_dist_thresholds = self.check_create_arr_for_hyperparam(wass_dist_thresholds)

        print('\n', '=='*10,' Start Segmentation ','=='*10,'\n')
        for i, ch in enumerate(channels):
            print(f' Channel: {ch}',)
            kde_win = None if kde_windows[i] == self.kde_windows[ch] else kde_windows[i]
            self.resegment_channel(ch,
                                   area_threshold= area_thresholds[i],
                                   width_threshold= width_thresholds[i],
                                   depth_threshold=depth_thresholds[i],
                                   kde_window = kde_win, verbose=verbose, init=True
                                   )
        self.commit_changes(verbose)

        if ref_samples_manual is None:
            print('\n', '==' * 10, ' Update Location Hierarchy and References', '==' * 10, '\n')
            self.recompute_and_update_location_hierarchy_and_refs(channels, jaccard_thresholds)

            self.commit_changes(verbose)
            print('\n', '=='*10,' Update Morphology Hierarchy and Morphology References','=='*10,'\n')

            self.update_morphology_hierarchy_and_refs(channels, wass_dist_thresholds)
        else:
            self.compute_location_morphology_groups_from_manual_refs(channels, ref_samples_manual, jaccard_thresholds, wass_dist_thresholds)

        self.commit_changes(verbose)

        print('\n', '=='*10,' Align Samples','=='*10,'\n')

        self.align_samples(channels, sigma=sigma, earth_smoothing_penalty=earth_smoothing_penalty,
                           n_sample=n_sample, subsample_ratio=subsample_ratio, verbose=verbose)
        self.commit_changes(verbose)


    def clean_and_export(self, path):

        '''

         Removes any Nan values, transforms data to the original input format and exports it to CSV file.

        :param path: :obj:`Str`, output path, including the csv file name.
        '''
        aligned_samples_data = []
        original_samples_data = []
        for s in range(self.num_samples):
            sample = []
            orig_sample = []
            for ch in range(self.num_channels):
                samp_ch = self.aligned_samples.reshape(-1, 1)
                sample +=[samp_ch]
            sample = np.concatenate(sample, axis=1 )
            y = np.zeros([sample.shape[0],1], dtype=np.int)+s
            sample = np.concatenate([y, sample], axis = 1)
            orig_sample = np.concatenate(orig_sample, axis=1 )
            orig_sample = np.concatenate([y, orig_sample], axis = 1)
            aligned_samples_data += [sample]
            original_samples_data += [orig_sample]

        aligned_samples_data = np.concatenate(aligned_samples_data)
        cleaned_and_aligned = aligned_samples_data[~np.isnan(aligned_samples_data).any(axis=1)]
        header = ','.join(self.channel_names)
        np.savetxt(path,cleaned_and_aligned,delimiter=',',header=header,
                   fmt=','.join(['%i'] + ['%1.4f']*self.num_channels), comments='')
        return cleaned_and_aligned

    def commit_changes(self, verbose=False):

        '''
        commits changes to the datahandler

        :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel for every sample.

        :ivar data_handler: :obj:`DataHandler`, a datahandler object that keeps track of changes to all pipeline modules.

        '''

        self.data_handler = DataHandler(self.samples, verbose=verbose)

class DataHandler:

    '''

    This is an abstraction class for dataset that facilitates certain queries on samples, channels, gates, groups, etc.

    :param  samples: :obj:`list`, a list of all samples in data:
    :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel for every sample.
    '''

    def __init__(self, samples, verbose=True):


        self.channels_gates = {}
        self.channels_gates_indexed_by_sample_num = {}
        self.channels_groups = {}
        self.samples = samples

        for ch in range(samples[0].num_ch):
            overall_gate_indx = 0
            for s, sample in enumerate(samples):
                gates = copy.copy(sample.gates[ch])
                num_gates = len(gates)
                for k, gate in enumerate(gates):
                    gate.overall_indx = k + overall_gate_indx

                if self.channels_gates.get(ch):
                   self.channels_gates[ch] += gates
                else:
                   self.channels_gates[ch] = gates

                overall_gate_indx += num_gates
            if verbose:
                print(self.channels_gates)
            self.channels_gates_indexed_by_sample_num[ch] = self.index_gates_by_sample_number(ch)

    def get_gates_of_channel(self, ch):

        '''

        Gets all gates in channel ch

        :param ch: :obj:`int`, number of channels.
        :returns:
         - channel gates - :obj:`list`, list of gates in channel ch.

        '''

        return self.channels_gates[ch]

    def get_gates_of_channel_dictionary_indexed_by_sample_num(self, ch):

        '''

        Gets all gates in channel ch indexed by sample number

        :param ch: :obj:`int`, number of channels.
        :returns:
         - :obj:`dict`, a dictionary of gates in channel ch indexed by sample number.

        '''

        return self.channels_gates_indexed_by_sample_num[ch]

    def index_gates_by_sample_number(self, ch):

        '''

        Indexes all gates in channel ch by sample number

        :param ch: :obj:`int`, number of channels.
        :returns:
         - :obj:`dict`, a dictionary of gates in channel ch indexed by sample number.

        '''

        gates_obj_arr = self.get_gates_of_channel(ch)

        gates_indx_dict = {}
        for i, g in enumerate(gates_obj_arr):
            samp_num = g.sample_num
            if gates_indx_dict.get(samp_num):
                gates_indx_dict[samp_num] += [i]
            else:
                gates_indx_dict[samp_num] = [i]
        return gates_indx_dict

    def get_groups_in_channel(self, ch, precomputed=False):

        '''

        :param ch: :obj:`int`, number of channels.
        :param precomputed: :obj:`Bool`, if :obj:`True`, saves computation time by returning previously computed version.
        :returns:
         - channels_groups - :obj:`dict`, a dictionary of gates in channel ch with location group as a get.

        '''

        if not precomputed:
            groups = []
            for gate in self.channels_gates[ch]:
                l_g = gate.location_group
                groups += [l_g]
            self.channels_groups[ch] = np.unique(groups)

        return self.channels_groups[ch]

    def get_two_groups_connection_weight(self,ch, g1, g2):

        '''

        Counts the number of samples that have gates with location groups g1 and g2.

        :param ch: :obj:`int`, channel number.
        :param g1: :obj:`int`, first location group.
        :param g2: :obj:`int`, second location group.
        :return: :obj:`int`, weight (number of shared samples).
        '''

        group_1_gates = self.get_gates_in_group(ch, g1)
        group_2_gates = self.get_gates_in_group(ch, g2)
        num_connected = 0
        for gate1 in group_1_gates:
            for gate2 in group_2_gates:
                if gate1.sample_num==gate2.sample_num:
                    # print('gr1',g1, 'gr2', g2, 'gate1_sample_num', gate1.sample_num, 'gate2.sample_num',gate2.sample_num)
                    # return 1
                    num_connected += 1
        return num_connected


    def get_groups_graph_matrix_of_ch(self, ch):

        '''
        Computes groups adjacency matrix in channel ch with weights (number of shared samples) between location groups as its enteries.

        :param ch: :obj:`int`, number of channels.
        :returns: :obj:`numpy.array(dtype=int)`, shape = [m, m], A adjacency matrix between groups.
        '''

        groups = self.get_groups_in_channel(ch)
        matrix = np.zeros(shape=[groups.shape[0], groups.shape[0]])
        for i,g1 in enumerate(groups):
            for j,g2 in enumerate(groups):
                matrix[i,j] = self.get_two_groups_connection_weight(ch, g1, g2)
        return matrix

    def get_gates_in_group(self, ch, group, group_type = 0):

        '''

        Gets all gates in location group :obj:`group` in channel ch.

        :param ch: :obj:`int`, number of channels.
        :param group: :obj:`int`, first location group.
        :param group_type: :obj:`int`,
        :return: gates: :obj:`list(Gate)`, a list of gates.
        '''

        result = list(filter(lambda gate:gate.location_group == group , self.channels_gates[ch]))
        return result

    def get_gates_and_location_groups_in_channel(self, ch):

        '''

        Gets a list of all gate objects in channel ch  as a pair (location group, gate)

        :param ch: :obj:`int`, number of channels.
        :return: gates: :obj:`list`, a list of pairs.
        '''

        groups = self.get_groups_in_channel(ch)
        result = []
        for g in groups:
            gates = self.get_gates_in_group(ch,g)
            result+= [ [g, gates]]
        return result

