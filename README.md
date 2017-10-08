# dibo
A web app which converts images to audio.

# Preliminaries

- How sound is represented mathematically and digitally?
    - Sound is a superposition of sine-waves, which is further sampled at some sampling rate.
- How to generate a *pleasant* sound using python?
    - Currently, focussing on piano music. `pysynth` seems to be a good library to synthesize piano music.
- Write an experiment to generate sound using a Fractal and growing circle traversing over it.
    - First requirement to do this is to come up with a mapping from an arbitrary-length array of integers to another fixed-length array of integers.

# TODOs

- Add superimpose feature to pysynth_b.
- Parallelize or improve time complexity of make_wav. [Done]
- Mapping from an arbitrary-length array of integers to another fixed-length array of fixed-integers.
- Make video of fractal music. [Done]