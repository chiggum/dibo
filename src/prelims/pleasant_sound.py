import pysynth

song = (
    ('c', 4),
    ('c*', 4), ('e', 4), ('g', 4),
    ('g*', 2), ('g5', 4),
    ('g5*', 4), ('r', 4), ('e5', 4),
    ('e5*', 4) 
)

pysynth.make_wav(song, fn = "song.wav")