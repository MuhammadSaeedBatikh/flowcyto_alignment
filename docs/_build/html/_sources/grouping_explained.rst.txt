.. _grouping_explained:

A Guide to Grouping and Ranking
================================

This section offers a simple guide to segments grouping and ranking and an explanation for the techniques used.

The Big Picture
======================


All the proposed approaches in the literature assume that the reference signal is a known sample. However, in practice, it is almost always the case that the reference signal is unknown, and we wish to automatically extract it from the dataset based on certain desiderata.
Given a dataset, we would like to group samples that have a particular distinct mode together.
This process is quite challenging for the following reasons:


Similarity Matrix
===========================

Given a set P_j = {f_(0,j) (x),f_(1,j) (x),…,f_(n,j) (x)} as the marginal pdfs of every sample for the jth channel where f_(i,j) (x) ∑_(k∈K_(i,j))▒〖w_(i,j,k ) P_(i,j,k) (x) 〗, and a collection of sets B_j={B_(0,j),B_(1,j),…,B_(n,j)} where B_(i,j) is the set of watershed lines present in the jth channel of the ith sample, and B_(i,j)={b_(i,j,0),b_(i,j,1),…,b_(i,j,m_(i,j) )  } where〖 b〗_(i,j,k) is the watershed line location and  m_(i,j) is the number of watershed lines. For each B_(i,j), we calculate a set gate G_(i,j)={g_(i,j,0),g_(i,j,1),…,g_(i,j,m_(i,j)-1)}  where  g_(i,j,k) is the pair (b_(i,j,k),b_(i,j,k+1) ). This pair represents the beginning and the end of the segment respectively. A more robust gate, is called a tightened gate which is computed using quantiles on both ends of segment’s pdf (mixture component) P_(i,j,k) (x) as shown in figure().

.. image:: ../src/imgs/Figure_1_samp.png
   :width: 30%
.. image:: ../src/imgs/Figure_1_samp.png
   :width: 30%
.. image:: ../src/imgs/Figure_2 open.png
   :width: 30%


Dendrogram
===============

Next, we compute Jaccard distance matrix M_j for all segments. Given a channel j, the number of segments in the channel is r=∏_i^n▒m_(i,j)   and 〖SM〗_j∈R^(r×r). The entries of M_j  are the Jaccard distance between each pair of segments. Jaccard Similarity, which is a standard metric for object detection tasks, is defined as intersection over union (IOU), J(A,B)=  (|A ∩ B|)/(|A ∪ B|) , which gives a number between 0 and 1 that determines the amount of overlapping between two boxes. In our case, given two segments locations A=(a_0,a_1) and B=(b_0,b_1 ), where a_0<a_1,b_0<b_1, |A ∩ B|=|max⁡(min⁡(a_1,b_1 )-max⁡(a_0,b_0 ),0) | and |A ∪ B|=|max⁡〖(a_1,b_1 )-〗  min⁡(a_0,b_0 ) |. The Jaccard distance is defined as d_j (A,B)=1-J(A,B). the Jaccard distance matrix for channel data shown in fig() is shown in figure().

.. image:: ../src/imgs/Figure_11.png
   :width: 70%
   :align: center


In order to obtain locations groups, we feed the precomputed Jaccard Distance Matrix to a hierarchical agglomerative clustering algorithm, which is a bottom-up hierarchical clustering approach, that starts with each segment as a separate cluster. Based on the precomputed Jaccard distance matrix M_j and a single-linkage creation, segments are grouped together

.. image:: ../src/imgs/Figure_10.png
   :width: 70%
   :align: center


Backtracking
===============

.. image:: ../src/imgs/Figure_1_counter.png
   :width: 30%
.. image:: ../src/imgs/Figure_2 tight.png
   :width: 30%
.. image:: ../src/imgs/Figure_2 figures.png
   :width: 30%



Deadlock
==========
