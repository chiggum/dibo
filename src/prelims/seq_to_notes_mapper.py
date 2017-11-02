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
    seq_ = seq.copy().flatten()
    kmeans = KMeans(n_clusters = 3, random_state=0).fit(seq_)
    return kmeans.labels_.reshape(seq.shape)

def map_category_sequence_to_notes(seq, map_type=0):
    if map_type == 0:
        assert seq.shape[0] == len(all_notes), \
            "No. of elements in a vector in the sequence\
            should be equal to the number of notes."
        # randomly shuffle all_notes
        all_notes_np = np.asarray(all_notes)
        np.random.shuffle(all_notes_np)
        # form notes sequence
        notes_seq = np.array(seq.shape, dtype=all_notes_np.dtype)
        for j in range(seq.shape[1]):
            for i in range(seq.shape[0]):
                if seq[i,j] == 0:
                    notes_seq[i,j] = None
                else:
                    notes_seq[i,j] = all_notes_np[i] + levels[seq[i,j]-1]
        return notes_seq