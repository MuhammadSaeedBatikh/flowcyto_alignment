from sklearn.preprocessing import MinMaxScaler
from src.cyto.segmentation import *
from src.cyto.grouping import *
from src.cyto.alignment import *
import time

class Pipeline:

    '''



        :param X_data:
        :param Y:
        :param num_channels:
        :param feature_range:
        :param channel_names:
        :param gate_factor_q:
        :param area_thresholds:
        :param kde_window:
        :param width_thresholds:
        :param depth_thresholds:
        :param jaccard_thresholds:
        :param wass_dist_threshold:
        :ivar min_before_min:
        :ivar max_before_norm:
        :ivar num_samples:
        :ivar num_channels:
        :ivar samples:
        :ivar Sim_Matrix_dict:
        :ivar agg_models_dict:
        :ivar location_groups_dict:
        :ivar Loc_Ref_Dict_All_Ch:
        :ivar incidence_matrices:
        :ivar incidence_matrices:
        :ivar loosened_groups_for_deadlock_dict:
        :ivar Morph_models_dict:
        :ivar Loc_Morph_Ref_Dict_All_Ch:
        :ivar Morph_groups_All_ch:
        :ivar wass_dist_threshold:
        :ivar aligned_samples:
        :ivar original_samples:
        :ivar gates_locations_dict:
        :ivar funcs_dict:
        :ivar comp_func_dict:
        :ivar earth_models_dict:

    '''
    def __init__(self, X_data, Y, num_channels, feature_range =(0,1), channel_names= None,
                            gate_factor_q=.02, area_thresholds=.04,
                            kde_window=.28, width_thresholds=.16,
                            depth_thresholds=.23, jaccard_thresholds=.6, wass_dist_threshold = 1e-9):
        # preprocessing and organizing

        self.max_before_norm, self.min_before_min = X_data.max(0), X_data.min(0)
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
        self.earth_models_dict = {}



    def segment_data(self, gate_factor_q=.02, area_thresholds=.04,
                            kde_window=.28, width_thresholds=.16,
                            depth_thresholds=.23, commit_changes= True, verbose= True):

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
                            depth_threshold=depth_thresholds)
            self.samples += [sample]

        time_end = time.time()
        print(time_end-time_start)

        if commit_changes:
            self.commit_changes(verbose)


    def resegment_channel(self, channel, area_threshold=None, width_threshold =None,
                                depth_threshold=None, kde_window=None, verbose= False):


        self.area_thresholds[channel]  = self.area_thresholds[channel]  if area_threshold  is None else area_threshold
        self.width_thresholds[channel] = self.width_thresholds[channel] if width_threshold is None else width_threshold
        self.depth_thresholds[channel] = self.depth_thresholds[channel] if depth_threshold is None else depth_threshold


        if kde_window is not None:
            self.kde_windows[channel] = kde_window

        for i, s in enumerate(self.samples):
            s.resegment_ch(channel, kde_window=kde_window, area_threshold=area_threshold,
                           width_threshold=width_threshold,
                           depth_threshold=depth_threshold,
                           init=False, verbose= verbose)

    def recompute_and_update_location_hierarchy_and_refs(self, channels, jaccard_thresholds):

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

    def align_samples(self, channels, sigma =1,  earth_smoothing_penalty=2, n_sample =-1, verbose= False):

        n_sample = self.num_samples if n_sample ==-1 else n_sample
        for ch in channels:
            print(channels, ch)
            align_samples_func(ch=ch, q_alignment = self.gate_factor_q[ch],
                               samples=self.samples, aligned_samples=self.aligned_samples,
                               original_samples=self.original_samples,
                               Loc_Ref_Dict_All_Ch=self.Loc_Ref_Dict_All_Ch,
                               Loc_Morph_Ref_Dict_All_Ch=self.Loc_Morph_Ref_Dict_All_Ch,
                               funcs_dict=self.funcs_dict, comp_func_dict=self.comp_func_dict,
                               gates_locations_dict=self.gates_locations_dict,
                               earth_models_dict=self.earth_models_dict,
                               n_sample = n_sample, sigma=sigma, earth_smoothing_penalty=earth_smoothing_penalty,
                               verbose = verbose)

        self.commit_changes(verbose)

    def clean_and_export(self, path):
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
        self.data_handler = DataHandler(self.samples, verbose=verbose)

class DataHandler:

    '''

    This is an abstraction class for dataset that facilitates certain queries on samples, channels, gates, groups, etc.

    :param  samples: :obj:`list`, a list of all samples in data:
    :param verbose: :obj:`Bool`, if :obj:`True`, prints all information about all gates in each channel for every sample.
    '''

    def __init__(self, samples, verbose=True):


        self.channels_gates = {}
        self.channels_groups = {}
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

    def get_gates_of_channel(self, ch):

        '''

        Gets all gates in channel ch

        :param ch: :obj:`int`, number of channels.
        :returns:
         - channel gates - :obj:`list`, list of gates in channel ch.

        '''

        return self.channels_gates[ch]

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
                    num_connected+=1
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

