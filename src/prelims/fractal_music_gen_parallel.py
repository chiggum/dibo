import numpy as np
import cv2
import argparse
import time
import multiprocessing
from pysynth import make_wav_par, make_wav_f
import subprocess
import os, sys
import matplotlib.cm as cm

from PyQt5 import QtCore, QtWidgets, QtGui

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
    def customColor(self, ind):
        mycol = self.widget.customColor(ind)
        return (mycol.red(), mycol.green(), mycol.blue())

app = QtWidgets.QApplication(sys.argv)
dialog = ColorDialog()
dialog.show()

vtoc = {}
cmap_sz = dialog.customCount()
iscmap_formed = False

def form_cmap(vals):
    vals = np.sort(vals)
    L = vals.shape[0]
    N = np.min([vals.shape[0], cmap_sz])
    vtoc[0.] = 0
    vtoc[1.] = N-1
    assert N > 1, "No. of fracs must be greater than 1."
    r_,g_,b_,a_ = cm.jet(0)
    dialog.setCustomColor(0,int(255*r_),int(255*g_),int(255*b_))
    r_,g_,b_,a_ = cm.jet(1)
    dialog.setCustomColor(N-1,int(255*r_),int(255*g_),int(255*b_))
    if N == 2:
        print("Only two different values found.")
    else:
        for i in range(1, N-1):
            frac = (1.0*i*(L-1))/(N-1)
            myclr = (frac/(L-1))
            r_,g_,b_,a_ = cm.jet(myclr)
            vtoc[vals[int(frac)]] = i
            dialog.setCustomColor(i,int(255*r_),int(255*g_),int(255*b_))

def get_customcolor_ind(val, vtoc_keys):
    i = len(vtoc_keys)
    while i > 0:
        if vtoc_keys[i-1] > val:
            i = i-1
        else:
            break
    return vtoc_keys[i-1]


def sym_icon_f(x,y,lmbda,alpha,beta,gamma,delta,omega,ndegree,pdegree):
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

def hit_pixel(x, y, scaleH, scaleW, H, W):
    xp = np.uint(x*scaleW*W + W/2.0 + 0.5)
    yp = np.uint(y*scaleH*H + H/2.0 + 0.5)
    return (xp,yp)

def get_param(category):
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

def get_img(hits, maxHits, H, W):
    global iscmap_formed
    if not iscmap_formed:
        form_cmap(np.power(0.45,(1.0*hits.flatten())/maxHits))
        iscmap_formed = True
    img = np.zeros((H,W,3),dtype=np.uint8)
    vtoc_keys = list(vtoc.keys())
    for i in range(H):
        for j in range(W):
            r_,g_,b_ = dialog.customColor(vtoc[get_customcolor_ind(np.power(0.45,(1.0*hits[i,j])/maxHits), vtoc_keys)])
            img[i,j,:] = (r_,g_,b_)
    return img

def iterate(x_init,y_init,category,
            scaleH, scaleW, H, W, n_iter):
    lmbda,alpha,beta,gamma,delta,omega,ndegree,pdegree = get_param(category)
    hits = np.zeros((H,W))
    maxHits = 1
    x_hit = x_init
    y_hit = y_init
    for it in range(n_iter):
        (x_hit, y_hit) = sym_icon_f(x_hit,y_hit,lmbda,alpha,beta,gamma,delta,omega,ndegree,pdegree)
        #print(x_hit,y_hit)
        (xp,yp) = hit_pixel(x_hit, y_hit, scaleH, scaleW, H, W)
        #print(xp,yp)
        if xp < W and yp < H:
            hits[yp,xp] += 1
            if hits[yp,xp] > maxHits:
                maxHits = hits[yp,xp]
    return (hits, maxHits)

if __name__=="__main__":
    all_notes = ['c1', 'c#1', 'd1', 'd#1', 'e1', 'f1', 'f#1', 'g1', 'g#1', 
                'a1', 'a#1', 'b1', 'c2', 'c#2', 'd2', 'd#2', 'e2', 'f2', 'f#2', 'g2', 'g#2',
                'a2', 'a#2', 'b2', 'c3', 'c#3', 'd3', 'd#3', 'e3', 'f3', 'f#3', 'g3', 'g#3',
                'a3', 'a#3', 'b3', 'c4', 'c#4', 'd4', 'd#4', 'e4', 'f4', 'f#4', 'g4', 'g#4',
                'a4', 'a#4', 'b4', 'c5', 'c#5', 'd5', 'd#5', 'e5', 'f5', 'f#5', 'g5', 'g#5',
                'a5', 'a#5', 'b5', 'c6', 'c#6', 'd6', 'd#6', 'e6', 'f6', 'f#6', 'g6', 'g#6',
                'a6', 'a#6', 'b6', 'c7', 'c#7', 'd7', 'd#7', 'e7', 'f7', 'f#7', 'g7', 'g#7',
                'a7', 'a#7', 'b7', 'c8']

    levels = ['', '*']

    def argument_parser():
        parser = argparse.ArgumentParser(description = 'description',
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-height', '--height', type=int,
                            help="Height of image",
                            default=len(all_notes))
        parser.add_argument('-width', '--width', type=int,
                            help="Width of image.",
                            required=True)
        parser.add_argument('-c', '--category', type=int,
                            help="Category of set.",
                            default=1)
        parser.add_argument('-f', '--ffmpeg_exe_path', type=str,
                            help="Category of set.",
                            default="G:/packages/ffmpeg/bin/ffmpeg.exe")
        parser.add_argument('-ni', '--n_iter', type=int,
                            help="No. of iterations.",
                            default=500000)
        parser.add_argument('-sh', '--scaleH', type=float,
                            help="scaleH.",
                            default=0.37)
        parser.add_argument('-sw', '--scaleW', type=float,
                            help="scaleW.",
                            default=0.37)
        parser.add_argument('-xi', '--x_init', type=float,
                            help="x_init.",
                            default=0.001)
        parser.add_argument('-yi', '--y_init', type=float,
                            help="y_init.",
                            default=0.002)
        return parser

    def parse_arguments(raw=None):
        args = argument_parser().parse_args(raw)
        return args

    args = parse_arguments()
    H = args.height
    # W = H   # Currently only supports square images
    W = args.width
    category = args.category
    scaleH = args.scaleH
    scaleW = args.scaleW
    n_iter = args.n_iter
    x_init = args.x_init
    y_init = args.y_init
    
    np.random.seed(2)
    N = 16
    myargs = []
    for i in range(N):
        x_init_ = x_init + np.random.normal(0,0.1)
        y_init_ = y_init + np.random.normal(0,0.1)
        print(x_init_,y_init_)
        myargs.append((x_init_, y_init_, category,
                        scaleH, scaleW, H, W, n_iter))
    
    with multiprocessing.Pool(processes=N) as pool:
        hits_maxhits_list = pool.starmap(iterate, myargs)
    
    hits_list = []
    maxhits_list = []
    maxHits = 0
    for h_,m_ in hits_maxhits_list:
        hits_list.append(h_)
        maxhits_list.append(m_)
        maxHits += m_
    hits = hits_list[0]*maxhits_list[0]
    for i in range(1,N):
        hits += hits_list[i]*maxhits_list[i]

    while True:
        h_mat = get_img(hits, maxHits, H, W)
        cv2.imshow("fractal", h_mat.reshape((H,W,3)))
        c = cv2.waitKey(0)
        if c == ord('f'):
            iReal-=realOffset
        elif c == ord('g'):
            iReal+=realOffset
        elif c == ord('v'):
            iImg -= imgOffset
        elif c == ord('b'):
            iImg += imgOffset
        elif c == ord('z') or c == ord('q'):
            break
        elif c == ord('s'):
            cv2.imwrite("fractal.png", h_mat.reshape((H,W,3)))
        elif c == ord('o'):
            offset -= 0.05*offMul
        elif c == ord('p'):
            offset += 0.05*offMul
        elif c == ord('k'):
            realMin -= 0.05*offMul
        elif c == ord('l'):
            realMin += 0.05*offMul
        elif c == ord('n'):
            imgMin -= 0.05*offMul
        elif c == ord('m'):
            imgMin += 0.05*offMul
        elif c == ord('u'):
            offMul/=10
        elif c == ord('i'):
            offMul*=10
        elif c == ord('r'):
            h_mat_ = h_mat.reshape((H,W,3))
            # make audio
            gs_mat = (.2126 * h_mat_[:,:,0] + .7152 * h_mat_[:,:,0] + .0722 * h_mat_[:,:,0]).astype(np.uint8)
            min_gs = np.min(gs_mat, axis=1).reshape((H,1))
            gs_mat -= min_gs
            gs_mat_sort = np.sort(gs_mat, axis=1)

            music_ = [[None]*W]*H
            for i in range(H):
                thresh1 = gs_mat_sort[i,int(W/3)]
                thresh2 = gs_mat_sort[i,int((2*W)/3)]
                for j in range(W):
                    if gs_mat[i,j] < thresh1:
                        pass
                    elif gs_mat[i,j] < thresh2:
                        music_[i][j] = (all_notes[i], 8)
                    else:
                        music_[i][j] = (all_notes[i]+'*', 8)
            
            par_args = []
            for j in range(W):
                temp = []
                for i in range(H):
                    if music_[i][j] is not None:
                        temp.append(music_[i][j])
                par_args.append((temp, True))
            
            t = time.time()
            with multiprocessing.Pool(processes=8) as pool:
                intensity_list_list = pool.starmap(make_wav_par, par_args)

            print("intensity lists prepared")
            print("Took", time.time()-t)
            t = time.time()
            fname = str(args.category) +  "_" + time.strftime("%d_%H_%M")
            make_wav_f(fname + ".wav", intensity_list_list)
            print("Took", time.time()-t)

            # make video
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(fname + ".avi", fourcc, 4.0, (W,H))
            for j in range(W):
                frame = h_mat_.copy()
                frame[:,j,:] = 255 - frame[:,j,:]
                out.write(frame)
            out.release()

            ## Use ffmpeg.exe -i audio_file -i video_file -c copy output.mkv
            cmd = args.ffmpeg_exe_path + " -i " + fname + ".avi -i " + fname + ".wav -c copy videos/" + fname + ".mkv"
            subprocess.run(cmd)

            # remove unnecessary files
            os.remove(fname+".avi")
            os.remove(fname+".wav")