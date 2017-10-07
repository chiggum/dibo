import pysynth

song = (
    ('c', 8),
    ('c*', 8), ('e', 8), ('g', 8),
    ('g*', 2), ('g5', 8),
    ('g5*', 8), ('r', 8), ('e5', 8),
    ('e5*', 8) 
)

traverse = [
    ('c', 8), ('c#', 8), ('d', 8), ('d#', 8), ('e', 8), ('f', 8), ('f#', 8), ('g', 8), ('g#', 8),
    ('a', 8), ('a#', 8), ('b', 8), 
    ('c', 8), ('c#', 8), ('d', 8), ('d#', 8), ('e', 8), ('f', 8), ('f#', 8), ('g', 8), ('g#', 8),
    ('a', 8), ('a#', 8), ('b', 8),
    ('c', 8), ('c#', 8), ('d', 8), ('d#', 8), ('e', 8), ('f', 8), ('f#', 8), ('g', 8), ('g#', 8),
    ('a', 8), ('a#', 8), ('b', 8),
]

traverse_unit = [
    ('c', 8), ('c#', 8), ('d', 8), ('d#', 8), ('e', 8), ('f', 8), ('f#', 8), ('g', 8), ('g#', 8),
    ('a', 8), ('a#', 8), ('b', 8)
]

# pysynth.make_wav(traverse, fn = "traverse.wav")

f_ = None
N=10
for i in range(N):
    if i == N-1:
        f_ = pysynth.make_wav(traverse_unit, fn = "traverse_overlap.wav", superimpose=True, close_f = True, f=f_)
    else:
        f_ = pysynth.make_wav(traverse_unit, fn = "traverse_overlap.wav", superimpose=True, close_f = False, f=f_)