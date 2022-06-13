from sklearn.cluster import AgglomerativeClustering
from simpleai.search import CspProblem, backtrack,min_conflicts, MOST_CONSTRAINED_VARIABLE, LEAST_CONSTRAINING_VALUE
from helpers import *
import copy



def build_similarity_matrix(gates, gate_q_factor = None, partial_columns = -1):
    '''

    Computes a matrix of Jaccard similarity indices between gates.

    :param gates: :obj:`list(Gate)`, :obj:`len(gates) = m` where m is the number of gates.
    :param gate_q_factor: :obj:`float`, tightening the gates using quantiles
    :param partial_columns: :obj:`list`, a list of gates indices to compute Jaccard similarity for across all other gates such that the result is a numpy.array`, :obj:`shape = [m, len(partial_columns)].(default = -1), creates square matrix and compares all gates.
                                if :obj:`None`: uses the original tight_gate in the gate object
                                if :obj:`-1`: uses open-ended gates
                                else: tightens the gates with the provided quantile value (slow).
    :return: :obj:`numpy.array`, :obj:`shape = [m, m]` A similarity matrix with Jaccard indices as entries.
    '''

    columns = list(range(len(gates))) if partial_columns ==-1 else partial_columns
    Sim_Matrix = np.ones(shape=(len(gates), len(columns)))
    print('here')
    for i in range(len(gates)):
        # this if-block is redundant to avoid computing quantiles multiple times for gate_A.
        if gate_q_factor is None:
            gate_A = gates[i].tight_gate
        elif gate_q_factor == -1:
            gate_A = gates[i].gate
        else:
            gate_A = gates[i].get_tight_gates(gate_q_factor)

        for jj, j in enumerate(columns):
            # if gates[i].sample_num != gates[j].sample_num:
            if i!=j:
                if gate_q_factor is None:
                    gate_B =  gates[j].tight_gate
                elif gate_q_factor == -1:
                    gate_B = gates[j].gate
                else:
                    gate_B =  gates[j].get_tight_gates(gate_q_factor)

                j_indx = jaccard(gate_A, gate_B)
                Sim_Matrix[i,jj] = j_indx
            # else:
            #     if i!=j:
            #         print('here')
            #         Sim_Matrix[i, j] = 0
    return Sim_Matrix


def build_distance_matrix(gates, n_samples = 1000):

    '''

    Creates a matrix with `morphology_distance` between each pair of gates as its entries.

    :param gates: :obj:`list(Gate)`, :obj:`len(gates) = m` where m is the number of gates.
    :param n_samples: :obj:`int`,number of cells sampled from each gate to compute the morphology_distance.
    :return: :obj:`numpy.array`, :obj:`shape = [m, m]`, distance_matrix.
    '''

    distance_matrix = np.zeros(shape=(len(gates), len(gates)))
    for i in range(len(gates)):
        A_seg = gates[i].segment
        ind = np.random.randint(0, A_seg.shape[0],
                                np.minimum(n_samples, A_seg.shape[0])
                                )
        A_seg_sub_sampled = A_seg[ind]
        for j in range(len(gates)):
            if gates[i].sample_num != gates[j].sample_num:
                B_seg =  gates[j].segment
                ind = np.random.randint(0, B_seg.shape[0],
                                        np.minimum(n_samples, B_seg.shape[0])
                                        )
                B_seg_sub_sampled = B_seg[ind]
                dist = morphology_distance(A_seg_sub_sampled, B_seg_sub_sampled)
                distance_matrix[i,j] = dist
    return distance_matrix

def group_gates(gates, dist_matrix, threshold, on_location= True):

    '''

    Based on the precomputed distance matrix and threshold value, a hierarchical clustering is performed to group gates together.
    It also updates group parameter in each gate.

    :param gates: :obj:`list(Gate)`, :obj:`len(gates) = m` where m is the number of gates.
    :param dist_matrix: :obj:`numpy.array`, shape = [m, m], A precomputed square distance matrix between each gate present in :obj:`gates`.
    :param threshold: :obj:`float`, the threshold value used to distinguish between different groups.
    :param on_location: :obj:`Bool`, default = :obj:`True`, if :obj:`True`, the method updates :obj:`location_group` in each gate object, otherwise it updates :obj:`morphology_group`.
    :return: :obj:`AgglomerativeClustering` agg_model and :obj:`np.array` groups as labels for each input gate.
    '''


    # 'Single linkage' is not coherent for it might allow separate gates in one sample to be members of the same location group too soon.
    # TODO: A combination of 'complete linkage' and punctuated Jaccard index should be used instead.

    agg_model =  AgglomerativeClustering(None,
                        affinity='precomputed',
                        distance_threshold = threshold,
                        linkage='single').fit(dist_matrix)
    groups = agg_model.labels_ +1
    for i, gate in enumerate(gates):
        if on_location:
            gate.location_group = groups[i]
        else:
            gate.morphology_group = groups[i]

    return agg_model, np.array(groups)


def binary_overlap_constarint(variables, values):

    '''

    Checks whether a pair of potential references overlap or not.

    :param variables: :obj:`tuple`, a tuple of groups present in a channel.
    :param values: :obj:`list`, a list of pairs of gates.
    :return: :obj:`Bool`, if :obj:`True`, the two gates overlaps.
    '''

    jind = jaccard(values[0].gate, values[1].gate)
    # check if jind within certain epislon
    # TODO: use precomputed Jac_Similarity Matrix
    if jind !=0:
        return False
    else:
        return True



def compute_constraints(incidence_matrix):
    '''

    Computes constraints from an incidence matrix.

    :param incidence_matrix :obj:`numpy.array`, an [m,m] weighted incidence matrix where m is the number of location groups in channel ch.
    :returns:
        - constraints - :obj:`list(tuple)`, a list of constraints. Each list element is a tuple([group 1, group2], function) such that the function is binary_overlap_constarint.

    '''
    groups_to_contstarint = np.argwhere(np.triu(incidence_matrix, k=1)) + 1

    constraints = []
    for gtc in groups_to_contstarint:
        constraints+= [ (gtc.tolist(), binary_overlap_constarint) ]
    return constraints


def get_references(ch, data_handler):

    '''

    Computes a set of references for channel ch that satisfy good reference desiderata, namely, maximally overlapping within group and non-overlapping between connected groups.
    This is the core method of this module. First, it computes groups incidence matrix in channel ch with weights representing the number of shared samples.
    Second, it sorts the data based on internal group reference scores which measure the goodness of reference for its own group each regardless of other considerations.
    Third, it builds a list of binary constraints based on connectivity between groups. Fourth, it searches the space for possible solutions using backtracking search algorithm, AC-3 arc consistency, and
    Least-Constraining-Value heuristics. Finally, it checks whether a deadlock has occurred and resolve the issue by loosening the group with the least important connections.

    :param ch: :obj:`int`, channel number.
    :param data_handler: :obj:`Datahandler`, a Datahandler object that facilitates certain quires on current data state.
    :returns:
      - result - :obj:`dict`, a dictionary of computed reference gate with location groups as keys.
      - incidence_matrix - :obj:`numpy.array`, an [m,m] weighted incidence matrix where m is the number of location groups in channel ch.
      - loosened_groups - :obj:`list`, a list of loosened groups in the case of a deadlock, empty otherwise.

    '''

    # getting groups and sorting gates based on their group reference scores
    groups = data_handler.get_groups_in_channel(ch)
    groups_gates = {gr:
                        sorted(data_handler.get_gates_in_group(ch, gr),
                               key= lambda gate: gate.group_ref_score, reverse=True)
                        for i, gr in enumerate(groups)
                    }

    # computing group incidence matrix
    incidence_matrix = data_handler.get_groups_graph_matrix_of_ch(ch)

    # CSP ingredients
    variables = tuple(groups)
    domains = groups_gates
    constraints = compute_constraints(incidence_matrix)
    incidence_matrix_cpy = copy.deepcopy(incidence_matrix)


    # my_problem = CspProblem(variables, domains, constraints)
    # initial_assignment = {k:gr[0] for k, gr in groups_gates.items()}
    #
    # result = min_conflicts(my_problem, initial_assignment=initial_assignment, iterations_limit=1000)
    # for r in  result.values():
    #     print((r.location_group,r.sample_num ,int(r.group_ref_score*100)/100))



    # optimal gates if we do not care about overlapping

    result = {key:value[0] for key, value in groups_gates.items()}
    print('optimal gates:')
    for r in  result.values():
        print((r.location_group, r.sample_num, int(r.group_ref_score*100)/100), r.gate)

    loosened_groups = []


    if len(constraints)<1:
        print('No Constraints')
    else:
        for i in range(incidence_matrix_cpy.shape[0]):
            print(f'\niter {i}', [c[0] for c in constraints])
            print(incidence_matrix_cpy)
            my_problem = CspProblem(variables, domains, constraints)
            result = backtrack(my_problem,
                                value_heuristic=LEAST_CONSTRAINING_VALUE,
                               inference=True)

            # Check if Deadlock and loosen the weakest constraint
            # TODO: This should be all the constraints of an entire group not a single constraint
            # TODO: flag gates from the group that has been ignored in order for the alignment module to treat it accordingly.

            if result is None:
                indx = np.argmin(incidence_matrix_cpy.sum(0))
                incidence_matrix_cpy[indx,:] = 0
                incidence_matrix_cpy[:,indx] = 0
                constraints = compute_constraints(incidence_matrix_cpy)
                loosened_groups += [indx+1]
            else:
                break

    # Flag gates in result as references.
    for v in result.values():
        v.is_location_reference = True

    # Flag gates in loosened groups to be handled by MARS.
    for group in loosened_groups:
        gates_in_loosened_group = data_handler.get_gates_in_group(ch,group)
        for gate in gates_in_loosened_group:
            gate.leave_to_MARS = True

    return result, incidence_matrix, loosened_groups


def assign_location_and_morphology_groups_from_JM(JM, ref_gates, targ_gates,
                                                  jaccard_threshold,
                                                  wass_dist_threshold, wass_n_samples):
    '''

    Givrn a Jaccard Similarity Matrix between a list of reference and target gates, this method assigns location and morphology groups for target gates and flags the rest for MARS model.

    :param JM: :obj:`numpy.array`, an [m,n] Jaccard similarity matrix for two samples where m is the number of reference gates and n is the number of target gates.
    :param ref_gates: :obj:`list(Gate)`: a list of reference gates.
    :param targ_gates: :obj:`list(Gate)` a list of target gates.
    :param jaccard_thresholds: :obj:`float`, Jaccard threshold indicating the minimum allowable Jaccard overlapping.
    :param wass_dist_threshold: obj:`float`.
    :param wass_n_samples: :obj:`int`, number of samples for computing wasserstien distance
    '''

    ll = len(targ_gates)
    leave_to_MARS_flags = [True]*ll

    for i in range(np.min(JM.shape)):
        # get most overlapping gate
        max_indx = JM.argmax()
        r, c = max_indx // JM.shape[1], max_indx % JM.shape[1]
        max_value = JM[r, c]
        if max_value >= jaccard_threshold:

            leave_to_MARS_flags[r] = False

            # Location
            targ_gates[r].location_group = c + 1
            JM[r, :], JM[:, c] = -1, -1


            # morphology
            z1 = ref_gates[c].segment
            z2 = targ_gates[r].segment
            ind1 = np.random.randint(0, z1.shape[0], np.minimum(wass_n_samples, z1.shape[0]))
            ind2 = np.random.randint(0, z2.shape[0], np.minimum(wass_n_samples, z2.shape[0]))

            wd = morphology_distance(z1[ind1], z2[ind2])
            if wd <= wass_dist_threshold:
                targ_gates[r].morphology_group = 1
        else:

            # even if the current maximum is higher than the threshold!
            break
    # flag the gates which do not have any references
    for i, flag in enumerate(leave_to_MARS_flags):
        targ_gates[i].leave_to_MARS = flag


def set_references_manually(channels, ref_samples,
                            data_handler, jaccard_thresholds,
                            wass_dist_thresholds,
                            Loc_Ref_Dict_All_Ch, Loc_Morph_Ref_Dict_All_Ch, wass_n_samples=2000
                            ):
    '''

    :param channels: :obj:`list(int)`, a list of channels to update.
    :param ref_samples: Union(:obj:`int`, :obj:`list`), number/s of reference sample/s for each channel.
    :param data_handler: :obj:`Datahandler`, a Datahandler object that facilitates certain quires on current data state.
    :param jaccard_thresholds: :obj:`dict`, a dictionary of Jaccard thresholds indicating where to cut the dendrogram with channels numbers as its keys.
    :param wass_dist_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing wasserstien distance thresholds for each channel.
    :param Loc_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of :obj:`Gate` chosen reference gates for all channels with channels numbers as its keys.
    :param Loc_Morph_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of :obj:`Gate` chosen morphology reference gates for all channels with (channel number, location group) tupel as its keys.
    :param wass_n_samples:
    '''

    # check input
    jaccard_thresholds = [jaccard_thresholds] * len(channels) if isinstance(jaccard_thresholds,
                                                                            float) else jaccard_thresholds
    wass_dist_thresholds = [wass_dist_thresholds] * len(channels) if isinstance(wass_dist_thresholds,
                                                                                float) else wass_dist_thresholds
    ref_samples = [ref_samples] * len(channels) if isinstance(ref_samples, int) else ref_samples

    for ind, ch in enumerate(channels):

        ref_sample = ref_samples[ind]
        ref_gates = data_handler.samples[ref_sample].gates[ch]
        jaccard_threshold, wass_dist_threshold = jaccard_thresholds[ind], wass_dist_thresholds[ind]

        gates_indx_dict = data_handler.get_gates_of_channel_dictionary_indexed_by_sample_num(ch)
        # build partial similarity matrix
        p_sim_matrix = build_similarity_matrix(data_handler.get_gates_of_channel(ch),
                                               partial_columns=gates_indx_dict[ref_sample])

        # assign location and morphology groups for every gate
        for samp_num, target_gates_indx in gates_indx_dict.items():
            print(samp_num, target_gates_indx)
            JM = p_sim_matrix[target_gates_indx, :]
            targ_gates = data_handler.samples[samp_num].gates[ch]
            assign_location_and_morphology_groups_from_JM(JM, ref_gates, targ_gates, jaccard_threshold,
                                                          wass_dist_threshold, wass_n_samples)

        # fill reference dictionaries
        Loc_Ref_Dict_All_Ch[ch] = {ref.location_group: ref for ref in ref_gates}
        Ref_Dict_indx_by_Loc_and_Morph = {}
        for group, ref in Loc_Ref_Dict_All_Ch[ch].items():
            ref.is_location_reference = True
            Ref_Dict_indx_by_Loc_and_Morph[group, 1] = ref

        Loc_Morph_Ref_Dict_All_Ch[ch] = Ref_Dict_indx_by_Loc_and_Morph


def recompute_and_update_location_hierarchy_and_refs(channels, data_handler, jaccard_thresholds,
                                    Sim_Matrix_dict, agg_models_dict, location_groups_dict, Loc_Ref_Dict_All_Ch,
                                                     incidence_matrices, loosened_groups_for_deadlock_dict):

    '''

     Recomputes references for the provided channels and updates Similarity Matrix Dictionary, Agglomerative Model Dictionary, Location Groups Dictionary,
     Location References Dictionary, Incidence Matrcies Dictionary, Loosened Groups For Deadlock Dictionary.

    :param channels: :obj:`list(int)`, a list of channels to update.
    :param data_handler: :obj:`Datahandler`, a Datahandler object that facilitates certain quires on current data state.
    :param jaccard_thresholds: :obj:`dict`, a dictionary of Jaccard thresholds indicating where to cut the dendrogram with channels numbers as its keys.
    :param Sim_Matrix_dict: :obj:`dict`, a dictionary of  [n,n] :obj:`numpy.array` Jaccard Similarity Matrices between pairs of gates with channels numbers as its keys where n is the number of gates.
    :param agg_models_dict: :obj:`dict`, a dictionary of computed :obj:`sklearn.cluster.AgglomerativeClustering` Agglomerative Models based on the computed Jaccard Similarity Matrices with channels numbers as its keys.
    :param location_groups_dict: :obj:`dict`, a dictionary of :obj:`int` location groups for each channel with channels numbers as its keys.
    :param Loc_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of :obj:`Gate` chosen reference gates for all channels with channels numbers as its keys.
    :param incidence_matrices: :obj:`dict`, a dictionary of  [m,m] :obj:`numpy.array` incidence matrices with channels numbers as its keys where m is the number of groups.
    :param loosened_groups_for_deadlock_dict: :obj:`dict`, a dictionary of constraints that have been loosened because of a deadlock with channels numbers as its keys.
    '''

    for ch in channels:
        print('ch',ch)
        gates_obj_arr = data_handler.get_gates_of_channel(ch)
        Sim_Matrix = build_similarity_matrix(gates_obj_arr)
        agg_model, location_groups =  group_gates(gates_obj_arr, 1-Sim_Matrix, 1-jaccard_thresholds[ch] )

        Sim_Matrix_dict[ch] = Sim_Matrix
        agg_models_dict[ch] = agg_model
        location_groups_dict[ch] =location_groups
        print('Unique groups', np.unique(location_groups),'\n'*2)
        Loc_Ref_Dict_All_Ch[ch], incidence_matrices[ch], loosened_groups_for_deadlock_dict[ch] = get_references(ch, data_handler)
        print(f'loosened_groups_for_deadlock_dict[{ch}]', loosened_groups_for_deadlock_dict[ch])


def update_morphology_hierarchy_and_refs(channels,
                                         wass_dist_threshold,
                                         data_handler,
                                         location_groups_dict,
                                         Loc_Morph_Ref_Dict_All_Ch,
                                         Morph_models_dict,
                                         Morph_groups_All_ch,
                                         n_samples = 2000
                                         ):
    '''

     Recomputes morphology references for the provided channels and updates Agglomerative Model Dictionary, Morphology Group Dictionary,
     and Morphology References Dictionary.

    :param channels: :obj:`list(int)`, a list of channels to update.
    :param wass_dist_threshold: :obj:`numpy.array(dtype=float)`, shape = [ch, 1], A 1-D array containing wasserstien distance thresholds for each channel.
    :param data_handler: :obj:`DataHandler`, an updated datahandler object.
    :param location_groups_dict: :obj:`dict`, a dictionary of :obj:`int` location groups for each channel with channels numbers as its keys.
    :param Loc_Morph_Ref_Dict_All_Ch: :obj:`dict`, a dictionary of :obj:`Gate` chosen morphology reference gates for all channels with (channel number, location group) tupel as its keys.
    :param Morph_models_dict: :obj:`dict`, a dictionary of computed :obj:`sklearn.cluster.AgglomerativeClustering` Agglomerative Models based on the computed Wasserstein Distance Matrices with channels numbers as its keys.
    :param Morph_groups_All_ch: :obj:`dict`, a dictionary of :obj:`int` morphology groups for each channel with channels numbers as its keys.
    :param n_samples: obj:`int`, number of cells used to compute the wasserstien distance, (default = 2000).
    '''

    for ch, location_groups in [[ch, location_groups_dict[ch]] for ch in channels]:
        distances = []
        print('\n','********\n'*4,'ch',ch)
        Ref_Dict_indx_by_Loc_and_Morph = {}
        Morph_models_list_ch = []
        Morph_groups_list = []
        for gr in np.unique(location_groups):
            gates_obj_arr = data_handler.get_gates_in_group(ch, gr)
            distance = build_distance_matrix(gates_obj_arr, n_samples = n_samples)

            print('\nloc_gr',gr,' #gates',len(gates_obj_arr),'\n')
            # print('dist:',distance,'\n')
            if distance.shape[0] >1:
                model, morph_groups = group_gates(gates_obj_arr, distance, wass_dist_threshold[ch], on_location=False)
                distances += [distance.flatten()]
                Morph_models_list_ch += [model]
                Morph_groups_list += [morph_groups]

                for m_group in np.unique(morph_groups):
                    indxes = np.argwhere(morph_groups==m_group).flatten()
                    if len(indxes)>1:
                        sub_matrix = distance[(morph_groups==m_group),:][:,(morph_groups==m_group)]
                        sub_indx = np.argmin(np.sum(sub_matrix, 0))
                        ref_indx = indxes[sub_indx]
                        Ref_Dict_indx_by_Loc_and_Morph[gr, m_group] = gates_obj_arr[ref_indx]
                        print('#members in group', len(indxes))
                        print(f'loc_gr:{gr}, morph_gr', m_group)
                        print('morph_groups', morph_groups)
                    print(f'indx of gates in morph_group {m_group}:', indxes)
                    # Morph_Ref_Dict_indx_by_Loc[] =
                    # Loc_Ref_Dict_Ch[group] =gates_obj_arr[ref_indx]
        Loc_Morph_Ref_Dict_All_Ch[ch] = Ref_Dict_indx_by_Loc_and_Morph
        Morph_models_dict[ch] = Morph_models_list_ch
        Morph_groups_All_ch[ch] = Morph_groups_list
