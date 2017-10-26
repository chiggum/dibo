import numpy as np
import cv2
import time
import multiprocessing
from pysynth import make_wav_par, make_wav_f
import subprocess
import os, sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets, QtGui
from utils import get_byte_to_note_map, get_hash, cluster_real_vals
from argparser import argument_parser, parse_arguments

"""
Custom color box to change the palette color.
"""
class ColorDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget = QtWidgets.QColorDialog()
        self.widget.setWindowFlags(QtCore.Qt.Widget)
        self.widget.setOptions(
            QtWidgets.QColorDialog.DontUseNativeDialog |
            QtWidgets.QColorDialog.NoButtons)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.widget)
    def customCount(self):
        return self.widget.customCount()
    def setCustomColor(self,ind,r,g,b):
        self.widget.setCustomColor(ind,QtGui.QColor(r,g,b))
    def customColor(self, ind1, ind2, w1, w2):
        mycol1 = self.widget.customColor(ind1)
        mycol2 = self.widget.customColor(ind2)
        return (int((mycol1.red()*w1 + mycol2.red()*w2)/(w1+w2)),
                int((mycol1.green()*w1 + mycol2.green()*w2)/(w1+w2)),
                int((mycol1.blue()*w1 + mycol2.blue()*w2)/(w1+w2)))

"""
Form color pallette using some heurisitc.
TODO: Find a better way to automatically
make a color pallette which looks good
when used in the image.
"""
def form_color_pallette(vals, args):
    print("Total number of different hits values:",vals.shape[0])
    vals = np.sort(np.unique(vals))
    L = vals.shape[0]
    print("Number of unique hits values:",L)
    N = np.min([L, args.dialog.customCount()])
    args.vtoc[0.] = 0
    args.vtoc[1.] = N-1
    assert N > 1, "No. of fracs must be greater than 1."
    r_,g_,b_,a_ = cm.hsv(0)
    args.dialog.setCustomColor(0,int(255*r_),int(255*g_),int(255*b_))
    r_,g_,b_,a_ = cm.hsv(1)
    args.dialog.setCustomColor(N-1,int(255*r_),int(255*g_),int(255*b_))
    if N == 2:
        print("Only two different values found.")
    else:
        for i in range(1, N-1):
            frac = (1.0*i*(L-1))/(N-1)
            myclr = (frac/(L-1))
            r_,g_,b_,a_ = cm.hsv(myclr)
            args.vtoc[vals[int(frac)]] = i
            args.dialog.setCustomColor(i,int(255*r_),int(255*g_),int(255*b_))

"""
Get index of custom color
with fractional values to
interpolate color.
"""
def get_customcolor_ind(val, vtoc_keys):
    i = len(vtoc_keys)-1
    while i > 0:
        if vtoc_keys[i-1] > val:
            i = i-1
        else:
            break
    return vtoc_keys[i-1], vtoc_keys[i], val - vtoc_keys[i-1], vtoc_keys[i] - val

"""
Symmetric icon parameters.
"""
def get_param_icon(args):
    category = args.category
    if category == 1:
        return -2.08,1.0,-0.1,0.167,0.0,0.0,7,0
    elif category == 2:
        return -2.7,5.0,1.5,1.0,0.0,0.0,6,0
    elif category == 3:
        return 2.5,-2.5,0,0.9,0,0,3,0
    elif category == 4:
        return 2.409,-2.5,0.0,0.9,0.0,0.0,23,0
    elif category == 5:
        return 2.5,-2.5,0.0,0.9,0.0,0.0,3,0
    elif category == 6:
        return 1.5,-1.0,-0.2,-0.75,0.04,0.0,3,24
    elif category == 7:
        return -2.05,3.0,-16.79,1.0,0.0,0.0,9,0
    return -2.08,1.0,-0.1,0.167,0.0,0.0,7,0

"""
Symmetric icon recursive function.
"""
def sym_icon_f(x,y,args):
    lmbda,alpha,beta,gamma,delta,\
    omega,ndegree,pdegree = get_param_icon(args)
    zzbar = x**2+y**2
    zzbarsqrt = np.sqrt(zzbar)
    x1 = 1
    y1 = 0
    x2 = 1
    y2 = 0
    for i in range(ndegree-1):
        x2 = x*x1 - y*y1
        y2 = x*y1 + x1*y
        x1 = x2
        y1 = y2
    xbar1 = x1
    ybar1 = -y1
    x2 = x*x1 - y*y1
    y2 = x*y1 + x1*y
    x1 = x2
    y1 = y2
    x4 = 0
    if delta != 0:
        x3 = x/zzbarsqrt
        y3 = y/zzbarsqrt
        x4 = 1
        y4 = 0
        for i in range(ndegree*pdegree):
            x2 = x3*x4 - y3*y4
            y2 = x3*y4 + x4*y3
            x4 = x2
            y4 = y2
    a = lmbda + alpha*zzbar + beta*x1 + delta*x4*zzbarsqrt
    return (a*x-omega*y+gamma*xbar1,a*y+omega*x+gamma*ybar1)

"""
Quilt parameters.
"""
def get_param_quilt(args):
    category = args.category
    if category == 1:
        return -0.2,-0.1,0.1,-0.25,0.0,0,0,0
    elif category == 2:
        return -0.59,0.2,0.1,-0.33,0.0,2,0,0
    elif category == 3:
        return 0.25,-0.3,0.2,0.3,0.0,1,0,0
    elif category == 4:
        return -0.12,-0.36,0.18,-0.14,0.0,1,0.5,0.5

"""
Quilt recursive function.
"""
def quilt_f(x,y,args):
    lmbda,alpha,beta,gamma,omega,m,v_x,v_y = get_param_quilt(args)
    x1 = m*x + v_x + lmbda*np.sin(2*np.pi*x) + alpha*np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + beta*np.sin(4*np.pi*x)+gamma*np.sin(6*np.pi*x)*np.cos(4*np.pi*y)
    y1 = m*y + v_y + lmbda*np.sin(2*np.pi*y) + alpha*np.sin(2*np.pi*y)*np.cos(2*np.pi*x) + beta*np.sin(4*np.pi*y)+gamma*np.sin(6*np.pi*y)*np.cos(4*np.pi*x)
    x1 = np.modf(x1)[0]
    x1 = np.modf(x1+1)[0]
    y1 = np.modf(y1)[0]
    y1 = np.modf(y1+1)[0]
    return (x1,y1)

"""
Fractal parameters.
"""
def get_param_fractal(args):
    category = args.category
    if category == 1:
        return -0.1,0.35,0.2,0.5,0.5,0.4

"""
Fractal recursive function.
"""
def fractal_f(x,y,args):
    a_11,a_12,a_21,a_22,b_1,b_2 = get_param_fractal(args)
    return (a_11*x+a_12*y+b_1, a_21*x+a_22*y+b_2)

"""
Hit this point.
"""
def hit_pixel(x,y,args):
    if args.which == 1:
        xp = np.uint(x*args.scale_width*args.width + args.width/2.0 + 0.5)
        yp = np.uint(y*args.scale_height*args.height + args.height/2.0 + 0.5)
        return (xp,yp)
    elif args.which == 2:
        xp = np.uint(x*args.patch_width + 0.5)
        yp = np.uint(y*args.patch_height + 0.5)
        return (xp,yp)
    elif args.which == 3:
        xp = np.uint(x*args.width + 0.5)
        yp = np.uint(y*args.height + 0.5)
        return (xp,yp)

"""
Computes hit map of pixels with the given
initial points.
"""
def iterate(x_init,y_init, args):
    hits = np.zeros((args.height, args.width))
    max_hits = 1
    x_hit = x_init
    y_hit = y_init
    for it in range(args.n_iter):
        if args.which == 1:
            (x_hit, y_hit) = sym_icon_f(x_hit,y_hit,args)
        elif args.which == 2:
            (x_hit, y_hit) = quilt_f(x_hit,y_hit,args)
        elif args.which == 3:
            (x_hit, y_hit) = fractal_f(x_hit,y_hit,args)
        (xp,yp) = hit_pixel(x_hit, y_hit, args)
        if (args.which == 1 and xp < args.width and yp < args.height)\
             or (args.which == 2 and xp < args.patch_width and yp < args.height)\
             or (args.which == 3 and xp < args.width and yp < args.height):
            hits[yp,xp] += 1
            if hits[yp,xp] > max_hits:
                max_hits = hits[yp,xp]
    return (hits, max_hits)

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
    for i in range(args.patch_height):
        for j in range(args.patch_width):
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
    hit_labels = cluster_real_vals(hits, args.cluster_prop)
    # prepare arguments for
    # parallel execution
    par_args = []
    for j in range(args.width):
        if args.which == 2:
            myval = np.asarray(hit_labels[:,j%args.patch_width], order='C')
        else:
            myval = np.asarray(hit_labels[:,j], order='C')
        myhash = get_hash(myval)
        i = 0
        temp = []
        while i < len(myhash):
            temp.append((byte_to_note_map[myhash[i:(i+2)]], 8))
            i = i + 2
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
    with multiprocessing.Pool(processes=args.num_proc) as pool:
        hits_maxhits_list = pool.starmap(iterate, par_args)
    # for sequential execution
    """
    hits_maxhits_list = []
    for i in range(args.num_proc):
        hits_, max_hits_ = iterate(par_args[i][0], par_args[i][1], args)
        hits_maxhits_list.append((hits_, max_hits_))
    """    
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