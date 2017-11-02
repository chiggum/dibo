import numpy as np
import cv2
import time
import multiprocessing
from pysynth import make_wav_par, make_wav_f
import subprocess
import os, sys
import matplotlib.pyplot as plt

from utils import get_byte_to_note_map, get_hash, cluster_real_vals
from argparser import argument_parser, parse_arguments
from PyQt5 import QtWidgets
from color_dialog import ColorDialog, form_color_pallette, get_customcolor_ind
from pattern_gen import iterate
from seq_to_notes_mapper import get_category_sequence, map_category_sequence_to_notes

"""
Get a colorful image using hits map
of pixels and some color pallette.
"""
def get_img(hits, maxHits, args):
    if not args.is_color_pallette_formed:
        form_color_pallette(hits.flatten()/maxHits, args)
        args.is_color_pallette_formed = True
    else:
        print("Color pallette already formed.")
    img = np.zeros((args.height, args.width, 3),dtype=np.uint8)
    if args.which == 2:
        patch_img = np.zeros((args.patch_height,args.patch_width,3),dtype=np.uint8)
    vtoc_keys = list(args.vtoc.keys())
    if args.which == 1:
        height_ = args.height
        width_ = args.width
    elif args.which == 2:
        height_ = args.patch_height
        width_ = args.patch_width
    elif args.which == 3:
        height_ = args.height
        width_ = args.width
    for i in range(height_):
        for j in range(width_):
            myval = (1.0*hits[i,j])/maxHits
            ind_prev, ind_next, w1, w2 = get_customcolor_ind(myval, vtoc_keys)
            r_,g_,b_ = args.dialog.customColor(args.vtoc[ind_prev], args.vtoc[ind_next], w1, w2)
            if args.which == 1:
                img[i,j,:] = (r_,g_,b_)
            elif args.which == 2:
                patch_img[i,j,:] = (r_,g_,b_)
            elif args.which == 3:
                img[i,j,:] = (r_,g_,b_)
    if args.which == 2:
        for i in range(int(args.height/args.patch_height)):
            for j in range(int(args.width/args.patch_width)):
                img[(args.patch_height*i):(args.patch_height*(i+1)),(args.patch_width*j):(args.patch_width*(j+1)),:] = patch_img
    return img

"""
Makes audio using pysynth and video
frames using opencv and pastes two
using ffmpeg to a file.
"""
def make_video(img, hits, byte_to_note_map, args):
    print("Making video...")
    print("#"*50)
    # make audio using pysynth
    # hit_labels = cluster_real_vals(hits, args.cluster_prop)
    hit_labels = get_category_sequence(hits)
    notes_seq = map_category_sequence_to_notes(hit_labels)
    # prepare arguments for
    # parallel execution
    par_args = []
    for j in range(args.width):
        if args.which == 2:
            #myval = np.asarray(hit_labels[:,j%args.patch_width], order='C')
            myval = notes_seq[:,j%args.patch_width]
        else:
            #myval = np.asarray(hit_labels[:,j], order='C')
            myval = notes_seq[:,j]
        #myhash = get_hash(myval)
        i = 0
        temp = []
        #while i < len(myhash):
        while i < myval.shape[0]:
            if myval[i] is not None:
                temp.append((myval[i], 8))
            i = i + 1
        par_args.append((temp, True))
    t = time.time()
    # go parallel
    with multiprocessing.Pool(processes=args.num_proc) as pool:
        ilist = pool.starmap(make_wav_par, par_args)
    print("Intensity lists prepared. Took", time.time()-t)
    t = time.time()
    fname = str(args.category) + "_" +\
            str(args.which) +  "_" +\
            time.strftime("%d_%H_%M")
    make_wav_f(fname + ".wav", ilist)
    print("Audio prepared. Took", time.time()-t)
    # make video using opencv
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # because audio is 4 fps
    out = cv2.VideoWriter(fname + ".avi", fourcc, 4.0, (args.width, args.height)) 
    for j in range(args.width):
        frame = img.copy()
        # invert jth line to indicate 
        # the pixels producing sound
        frame[:,j,:] = 255 - frame[:,j,:]
        out.write(frame)
    out.release()
    ## paste audio + video using ffmpeg
    cmd = args.ffmpeg_exe_path + " -i " + \
            fname + ".avi -i " + fname + \
            ".wav -c copy videos/" + fname + ".mkv"
    subprocess.run(cmd)
    # remove unnecessary files
    os.remove(fname+".avi")
    os.remove(fname+".wav")

"""
Computes the hits map of pixels by calling iterate
over several initial points (=args.num_proc)
and aggregate results to get final hits 
map and max number of hits
"""
def get_hits_map(args):
    print("Computing hits map...")
    print("#"*50)
    par_args = []
    for i in range(args.num_proc):
        if args.which == 1:
            x_init_ = args.x_init + np.random.normal(0,0.1)
            y_init_ = args.y_init + np.random.normal(0,0.1)
        elif args.which == 2:
            x_init_ = np.modf(1+np.modf(np.random.normal(0,1))[0])[0]
            y_init_ = np.modf(1+np.modf(np.random.normal(0,1))[0])[0]
        elif args.which == 3:
            x_init_ = np.modf(1+np.modf(np.random.normal(0,1))[0])[0]
            y_init_ = np.modf(1+np.modf(np.random.normal(0,1))[0])[0]
        par_args.append((x_init_, y_init_, args))
    
    # for parallel execution
    if args.parallel_hits_map:
        with multiprocessing.Pool(processes=args.num_proc) as pool:
            hits_maxhits_list = pool.starmap(iterate, par_args)
    # for sequential execution
    else:
        hits_maxhits_list = []
        for i in range(args.num_proc):
            hits_, max_hits_ = iterate(par_args[i][0], par_args[i][1], args)
            hits_maxhits_list.append((hits_, max_hits_))
       
    # aggregate hits and max_hits
    hits_list = []
    max_hits_sum = 0
    maxhits_list = []
    for h_,m_ in hits_maxhits_list:
        hits_list.append(h_)
        maxhits_list.append(m_)
        max_hits_sum += m_
    hits = hits_list[0]*maxhits_list[0]
    for i in range(1,args.num_proc):
        hits += hits_list[i]*maxhits_list[i]
    hits = (1.0*hits)/max_hits_sum
    max_hits = np.max(hits)
    return hits, max_hits
            
if __name__=="__main__":
    args = parse_arguments()

    hits, max_hits = get_hits_map(args)
    # plt.imshow(hits/max_hits, cmap='jet')
    # plt.show()

    args.is_color_pallette_formed = False
    app = QtWidgets.QApplication(sys.argv)
    args.dialog = ColorDialog()
    args.vtoc = {}
    args.dialog.show()

    while True:
        # get a colorful image of 
        # fractal/icon/quilt from hits
        img = get_img(hits, max_hits, args)
        cv2.imshow("fractal", img)
        # wait for user to press key
        c = cv2.waitKey(0)
        if c == ord('z') or c == ord('q'):  # exit
            break
        elif c == ord('s'): # save image
            fname = str(args.category) + "_" + \
                    str(args.which) +  "_" + \
                    time.strftime("%d_%H_%M")
            cv2.imwrite("images/"+fname+".png", img)
        elif c == ord('r'): # make video
            byte_to_note_map = get_byte_to_note_map() 
            make_video(img, hits, byte_to_note_map, args)