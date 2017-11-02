import hashlib, xxhash
import numpy as np

all_notes = ['c1', 'c#1', 'd1', 'd#1', 'e1', 'f1', 'f#1', 'g1', 'g#1', 
                'a1', 'a#1', 'b1', 'c2', 'c#2', 'd2', 'd#2', 'e2', 'f2', 'f#2', 'g2', 'g#2',
                'a2', 'a#2', 'b2', 'c3', 'c#3', 'd3', 'd#3', 'e3', 'f3', 'f#3', 'g3', 'g#3',
                'a3', 'a#3', 'b3', 'c4', 'c#4', 'd4', 'd#4', 'e4', 'f4', 'f#4', 'g4', 'g#4',
                'a4', 'a#4', 'b4', 'c5', 'c#5', 'd5', 'd#5', 'e5', 'f5', 'f#5', 'g5', 'g#5',
                'a5', 'a#5', 'b5', 'c6', 'c#6', 'd6', 'd#6', 'e6', 'f6', 'f#6', 'g6', 'g#6',
                'a6', 'a#6', 'b6', 'c7', 'c#7', 'd7', 'd#7', 'e7', 'f7', 'f#7', 'g7', 'g#7',
                'a7', 'a#7', 'b7']

levels = ['', '']

"""
Map each byte to a note. Currently, bytes are mapped
to a note without any rule.
TODO: Make a map which can result in a melody.
Some thoughts:
"""
def get_byte_to_note_map(keys = ['a','b','c','d','e','f','g'],
                        levels = ['','#'],
                        octaves = ['1','2','3','4','5','6','7'],
                        intensities = ['','*'],
                        max_cnt = 255):
    byte_to_note_map = {}
    cnt = 0
    max_byte_int = 0
    for elem1 in keys:
        for elem2 in levels:
            for elem3 in octaves:
                for elem4 in intensities:
                    mynote = elem1+elem2+elem3+elem4
                    mybyte = "%02x" % ord((cnt).to_bytes(1,'big'))
                    byte_to_note_map[mybyte] = mynote
                    cnt += 1
    max_byte_int = cnt
    while cnt <= max_cnt:
        mybyte = "%02x" % ord((cnt).to_bytes(1,'big'))
        mybyte2 = "%02x" % ord((cnt%max_byte_int).to_bytes(1,'big'))
        byte_to_note_map[mybyte] = byte_to_note_map[mybyte2]
        cnt += 1
    return byte_to_note_map

"""
Computes a fixed length hash of a sequence (of pixels).
TODO: Replace with a non random hash which depends
on sequence of ints. The sequence length can be
arbitrary but the hash length should be fixed.
"""
def get_hash(val):
        # return hashlib.sha256(val).hexdigest()
        return xxhash.xxh64(val).hexdigest()

"""
Clusters a nD array of real numbers.
Make sure that 1/prop is a natural number
and is not greater than 255.
Number of clusters = 1/prop
"""
def cluster_real_vals(vals, prop=0.1):
        a = np.sort(vals.flatten())
        L = a.shape[0]
        labels = np.zeros(vals.shape, dtype=np.uint8)
        tot_prop = 0.
        label = 0
        last_val = a[0]-1
        while tot_prop < 1:
            u_ind = np.min([L, int((tot_prop+prop)*L)])
            u_val = a[u_ind-1]
            labels[np.logical_and(vals > last_val, vals <= u_val)] = label
            last_val = u_val
            label += 1
            tot_prop += prop
        return labels