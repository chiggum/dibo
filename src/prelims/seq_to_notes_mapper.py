import numpy as np
from sklearn.cluster import KMeans
from utils import all_notes, levels

"""
Clusters values in a sequence of real vectors 
provided in the form of numpy nD array.
Returns a sequence of labels of same shape as
input sequence.
"""
def get_category_sequence(seq, num_clusters=3):
    seq_ = seq.copy().reshape(-1,1)
    kmeans = KMeans(n_clusters = num_clusters, random_state=0).fit(seq_)
    return kmeans.labels_.reshape(seq.shape)

def map_category_sequence_to_notes(seq, map_type=0):
    if map_type == 0:
        assert seq.shape[0] == len(all_notes),\
            "No. of elements in a vector in the sequence\
            should be equal to the number of notes. Got"\
            + str(seq.shape[0]) + " vs " + str(len(all_notes))
        # form notes sequence
        notes_seq = []
        for j in range(seq.shape[1]):
            this_notes_seq = []
            for i in range(seq.shape[0]):
                if seq[i,j] != 0:
                    this_notes_seq.append(all_notes[i] + levels[seq[i,j]-1])
        return notes_seq
    elif map_type == 1:
        assert (seq.shape[0]**2 + seq.shape[1]**2) <= 4*len(all_notes)*len(all_notes),\
            "No. of elements in diagonal in the sequence\
            should be less than twice the number of notes. Got"\
            + str(seq.shape[0]**2 + seq.shape[1]**2) + " vs " + str(len(all_notes))
        max_degrees = 360
        notes_seq = []
        centre_h = seq.shape[0]/2
        centre_w = seq.shape[1]/2
        for theta in range(max_degrees):
            this_degree_notes = []
            (i_, j_) = centre_h, centre_w
            radius = 0
            while i_ >= 0 and i_ < seq.shape[0] and j_ >= 0 and j_ < seq.shape[1]:
                if seq[int(i_), int(j_)] != 0:
                    this_degree_notes.append(all_notes[radius] + levels[seq[int(i_), int(j_)]-1])
                radius += 1
                i_ = i_ + np.cos(theta*np.pi/180.)
                j_ = j_ + np.sin(theta*np.pi/180)
            notes_seq.append(this_degree_notes)
        return notes_seq