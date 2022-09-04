import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import networkx as nx
import seaborn as sns
import functools

colors = 'blue,orange,g,r,coral,cyan,m,steelblue,brown,y,gray,k'.split(',')



def plot_gates_dendrogram(agg_model, ch, jacc_threshold,  column_name='', save=False, show_title=True, gatefontsize=13, yfontsize=16):

        '''

        :param agg_model:
        :param ch:
        :param jacc_threshold:
        :param column_name:
        :param save:
        :param show_title:
        :param gatefontsize:
        :param yfontsize:
        :return:
        '''


        fig = plt.figure()
        fig.set_size_inches(28.5, 10.5)
        if show_title:
            fig.suptitle(f'Ch: {ch} {column_name}',fontsize=30)
        plt.axhline(1 - jacc_threshold)
        plot_dendrogram(agg_model)
        plt.xticks(fontsize= gatefontsize)
        plt.yticks(fontsize= yfontsize)
        fig.tight_layout()
        if save:
            fig.savefig(f'../imgs/location_dendrogram_ch_{ch}')


def plot_original_vs_aligned(channels,samples, aligned_samples,
                            Loc_Ref_Dict_All_Ch, columns_names,n_sample = -1,
                             max_per_plt=15, root_path = None, xlim0=0, xlim1 = 1,):


    '''

    :param channels:
    :param samples:
    :param aligned_samples:
    :param n_sample:
    :param Loc_Ref_Dict_All_Ch:
    :param columns_names:
    :param max_per_plt:
    :param root_path:
    :param xlim0:
    :param xlim1:
    :return:
    '''

    n_sample = len(samples) if n_sample == -1 else n_sample
    for ch in channels:
        fig = None
        part = -1
        for i in range(n_sample):
            if i%max_per_plt==0:
                if fig is not None and root_path is not None:
                    path = f'{root_path}/aligned_ch_{ch}_part_{part}'
                    fig.savefig(path)
                fig, axes = plt.subplots(np.minimum(n_sample, max_per_plt), 2)
                part+=1
            fig.suptitle(f'Ch:{ch} {columns_names[ch+1]}, Samples: {part*max_per_plt} -> {np.minimum((part+1)*max_per_plt-1, n_sample-1)}')
            samp = samples[i]
            sns.distplot(samp(ch), ax=axes[i%max_per_plt,0],hist=True,bins=128, label=f'orig:{i}')
            lo_gr = samp.gates[ch][0].location_group
            ref_gate = Loc_Ref_Dict_All_Ch[ch][lo_gr]

            z = aligned_samples[i, ch]
            b = False
            for gate in samp.gates[ch]:
                lo_gr = gate.location_group
                ref_gate = Loc_Ref_Dict_All_Ch[ch][lo_gr]
                if functools.reduce(lambda x,y: x | (y==ref_gate),samp.gates[ch],False):
                    b = True
                    break
            if b:
                sns.distplot(z ,ax= axes[i%max_per_plt,1], hist= False, color = colors[lo_gr+4])
            else:
                sns.distplot(z, ax= axes[i%max_per_plt,1], hist= True, bins=128, color = colors[lo_gr+4])

            axes[i%max_per_plt,0].set_xticks([])
            axes[i%max_per_plt,0].set_yticks([])
            axes[i%max_per_plt,1].set_xticks([])
            axes[i%max_per_plt,1].set_yticks([])
            axes[i%max_per_plt,0].set_ylabel(' ')
            axes[i%max_per_plt,0].set_xlim(xlim0,xlim1)
            axes[i%max_per_plt,1].set_xlim(xlim0,xlim1)
            axes[i%max_per_plt,1].set_ylabel('')

            if i == n_sample-1 and root_path is not None:
                path = f'{root_path}/aligned_ch_{ch}_part_{part}'
                fig.savefig(path)

def plot_channel_segments(ch:int, samples:list, max_per_plt=12, color_fill_lg =False,
                          show_labels=True, limit_x0=-.1,  limit_x1=1.1):
    '''

    Given a list of samples and a channel number, this method plots the channel marginal probability density function for each sample indicating the location of segments/gates.

    :param ch: :obj:`int`, channel number.
    :param samples: :obj:`list(Sample)`, list of :obj:`Sample` objects.
    :param max_per_plt: :obj:`int`, maximum number of plots allowed per plot.
    :param limit_x0: :obj:`float`, plotting start point.
    :param limit_x1: :obj:`float`, plotting end point.
    :param color_fill_lg: :obj:`Bool`, default = :obj:`False`, if :obj:`True`, colors the gate/segment according to its location group.
    :param show_labels: :obj:`Bool`, default = :obj:`True`, if :obj:`True`, the group labels are shown on each gate/segment.
    '''

    for s, sample in enumerate(samples):
        if s%max_per_plt ==0:
            axes = plt.subplots(np.minimum(len(samples), max_per_plt),1)[1]

        # sns.distplot(samples[s](ch), ax=axes[s%max_per_plt],hist=True,bins=100,label=f'{s}',
        #              kde_kws={'bw_method': sample.kde_window[ch]})
        x, pdf = sample.pdfs[ch]
        if not color_fill_lg:
            axes[s % max_per_plt].fill(x, pdf,label=f'{s}')

        axes[s%max_per_plt].axis('off')
        axes[s%max_per_plt].set_xticks([])
        axes[s%max_per_plt].set_yticks([])
        axes[s%max_per_plt].set_ylabel('')
        axes[s%max_per_plt].set_xlim(limit_x0, limit_x1)

        for g_obj in sample.gates[ch]:
            a,b = g_obj.gate[0], g_obj.gate[1]
            axes[s%max_per_plt].axvline(a, c='k')
            axes[s%max_per_plt].axvline(b, c='k')

            if color_fill_lg:
                lr = g_obj.location_group
                alpha = .3 if g_obj.is_location_reference else 1
                axes[s % max_per_plt].fill_between(x, pdf,
                                                   where=(x >= a) & (x < b),
                                                   label=f'{s}',
                                                   alpha = alpha,
                                                   color=colors[lr+2])


            if show_labels:
                gate_title = f'{g_obj.overall_indx}: (loc:{g_obj.location_group}, morph:{g_obj.morphology_group})'
                axes[s%max_per_plt].text(a + (b-a)/2 , g_obj.pd.mean() , gate_title,c='k', size=12)
    plt.show()

def plot_dendrogram(model, **kwargs):
    '''

    Create linkage matrix and then plot the dendrogram.

    :param model: :obj:`sklearn.cluster.AgglomerativeClustering`: Agglomerative Clustering model which contains necessary attributes regarding groups, hierarchy and so forth.
    :param kwargs: :obj:`dict` keyword arguments passed to the :obj:`scipy.cluster.hierarchy.dendrogram` method.
    '''
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    return dendrogram(linkage_matrix, **kwargs)

def plot_gate_as_tiles(data_handler, ch, Loc_Ref_Dict_All_Ch= None, color_tiles=True, linewidth=10,rev= -1,  shift_color=2, show_ref=True, tight=True, show_num=False):
    '''

    Plot all samples gates of a given channel as tiles with colors indicating their location groups.

    :param data_handler: :obj:`DataHandler` object.
    :param ch: :obj:`int`, channel number.
    :param linewidth: :obj:`float`, width of each line/tile.
    :param rev: :obj:`int`, either [-1,1], -1 reverse the order of tiles.
    :param shift_color: :obj:`int`, shift color number from colors palette.
    :param show_ref: :obj:`Bool`, if True, draws a black line in the middle of the tile corresponding to reference gate.
    '''
    counter = -1
    gates_list = data_handler.get_gates_of_channel(ch)
    for k, gate in enumerate(gates_list):
        counter+=1

        s_num, location_group = gate.sample_num, gate.location_group
        if not color_tiles:
            location_group=gate.gate_num_in_sample
        flag = False
        if Loc_Ref_Dict_All_Ch is not None:
            ref = Loc_Ref_Dict_All_Ch[ch][location_group]
            #Todo: fix gate comparison. (__eq__)
            flag = ref.sample_num  == gate.sample_num and ref.gate[0] == gate.gate[0] and ref.gate[1] == gate.gate[1]
        a, b = gate.tight_gate if tight else  gate.gate
        t = np.linspace(a, b, 20)
        if show_num:
            plt.text(t.mean(), rev* s_num + linewidth/100, f'{counter}', color='k', fontsize=12)

        plt.plot(t, np.zeros(t.shape) + rev* s_num, color=colors[location_group + shift_color%len(colors)], linewidth=linewidth,zorder=1)

        if flag and show_ref:
            plt.plot(t,np.zeros(t.shape) +rev* s_num, color='k', linewidth=linewidth/5,zorder=2)


    plt.yticks([])

def plot_lines(x,y, n_lines, linewidth = .6,s1=10, s2=20, ax =None):

    '''

    :param x:
    :param y:
    :param n_lines:
    :param linewidth:
    :param s1:
    :param s2:
    :param ax:
    :return:
    '''

    ax = plt.subplots(1,1)[1] if ax is None else ax
    s = x.shape[0]//n_lines
    for i in range(0, x.shape[0], s):
        l0 = [x[i,0], y[i,0]]
        l1 = [x[i,1], y[i,1]]
        ax.plot(l0,l1, c = 'k', linewidth=linewidth,)
        # ax.arrow(*l0, *l1,
        #          length_includes_head=True,head_width=0.03)
    ax.scatter(x[:,0],x[:,1], s= s1, alpha=.5, c='b')
    ax.scatter(y[:,0],y[:,1], s= s2, alpha=.5, c='g')

def plot_network_graph(ch, incidence_matrix, columns_names=None):

    '''

    :param ch:
    :param incidence_matrix:
    :param columns_names:
    :return:
    '''

    np.random.seed(3)
    fig, ax = plt.subplots(1,1)
    if columns_names is not None:
        ax.set_title(f'{ch}: {columns_names[ch+1]}')
    G = nx.Graph()
    groups = (np.argwhere(np.triu(incidence_matrix, k=0))).tolist()
    print(groups)
    for g in groups:
        w = incidence_matrix[g[0], g[1]]
        G.add_edge(f'{g[0]+1}',f'{g[1]+1}',weight=w)

    pos = nx.spring_layout(G)
    s = 2
    nx.draw(G, pos,ax=ax, node_size=600, node_color=colors[s%colors.__len__():(incidence_matrix.shape[0]+s)%colors.__len__()])
    nx.draw_networkx_labels(G, pos)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def plot_scatter_aligned_vs_original(channels,pipeline=None, samples=None, aligned_samples= None, columns_names= None):

    '''

    :param channels:
    :param pipeline:
    :param samples:
    :param aligned_samples:
    :param columns_names:
    :return:
    '''


    for ch1, ch2 in channels:

        print(columns_names[ch1 + 1], columns_names[ch2 + 1])
        axes = plt.subplots(1, 2)[1]
        for s in range(0, 60):
            if pipeline is not  None:
                axes[0].scatter(pipeline.samples[s](ch1), pipeline.samples[s](ch2), s=.002, c='b')
                axes[1].scatter(pipeline.aligned_samples[s, ch1], pipeline.aligned_samples[s, ch2], s=.002, c='b')
            else:
                axes[0].scatter(samples[s](ch1), samples[s](ch2), s=.002, c='b')
                axes[1].scatter(aligned_samples[s, ch1], aligned_samples[s, ch2], s=.002, c='b')