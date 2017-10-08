import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import cv2
import argparse
import time
import multiprocessing
from pysynth import make_wav_par, make_wav_f
from pycuda.compiler import SourceModule

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
        return parser

    def parse_arguments(raw=None):
        args = argument_parser().parse_args(raw)
        return args

    # Configuration:
    interceptRealMin = -1
    interceptRealMax = 1
    interceptImgMin = -1
    interceptImgMax = 1
    realOffset = 0.01
    imgOffset = 0.01
    maxZAbs = 100
    maxN = 255
    minN = 5
    decayN = 5
    offset = 4
    realMin = -2
    imgMin = -2	
    offMul = 1
    numColors = 52
    BLOCKSIZE = 512

    kernel_code = """
    #include <stdint.h>
    #include <cuComplex.h>
    #include <math.h>
    #define CU_PI 3.141592654f
    /*********************************************
    Custom functions
    *********************************************/
    __device__ cuDoubleComplex cuCexp(cuDoubleComplex z) {
        double z_r = cuCreal(z);
        double z_i = cuCimag(z);
        cuDoubleComplex ret = make_cuDoubleComplex(exp(z_r)*cos(z_i), exp(z_r)*sin(z_i));
        return ret;
    }

    __device__ cuDoubleComplex cuClog(cuDoubleComplex z) {
        double z_mod = cuCabs(z);
        double z_theta;
        double z_r = cuCreal(z);
        double z_i = cuCimag(z);
        if(z_r == 0) {
            if(z_i >= 0) {
                z_theta = CU_PI*0.5;
            } else {
                z_theta = -CU_PI*0.5;
            }
        } else {
            z_theta = atan(z_i/z_r);
        }
        cuDoubleComplex ret = make_cuDoubleComplex(log(z_mod), z_theta);
        return ret;
    }

    __device__ cuDoubleComplex cuCpow(cuDoubleComplex z, cuDoubleComplex y) {
        double z_r = cuCreal(z);
        double z_i = cuCimag(z);
        cuDoubleComplex z_log = cuClog(z);
        cuDoubleComplex ret = cuCexp(cuCmul(y,z_log));
        return ret;
    }

    __device__ cuDoubleComplex cuCcos(cuDoubleComplex z) {
        cuDoubleComplex ret = cuCdiv(cuCadd(cuCexp(cuCmul(make_cuDoubleComplex(0,1),z)),cuCexp(cuCmul(make_cuDoubleComplex(0,-1),z))), make_cuDoubleComplex(2,0));
        return ret;
    }

    __device__ cuDoubleComplex cuCsin(cuDoubleComplex z) {
        cuDoubleComplex ret = cuCdiv(cuCadd(cuCexp(cuCmul(make_cuDoubleComplex(0,1),z)),cuCmul(make_cuDoubleComplex(-1,0),cuCexp(cuCmul(make_cuDoubleComplex(0,-1),z)))), make_cuDoubleComplex(0,2));
        return ret;
    }

    __device__ cuDoubleComplex cuCcosh(cuDoubleComplex z) {
        cuDoubleComplex ret = cuCdiv(cuCadd(cuCexp(z),cuCexp(cuCmul(make_cuDoubleComplex(-1,0),z))), make_cuDoubleComplex(2,0));
        return ret;
    }

    __device__ cuDoubleComplex cuCsinh(cuDoubleComplex z) {
        cuDoubleComplex ret = cuCdiv(cuCadd(cuCexp(z),cuCmul(make_cuDoubleComplex(-1,0),cuCexp(cuCmul(make_cuDoubleComplex(-1,0),z)))), make_cuDoubleComplex(2,0));
        return ret;
    }
    /*********************************************
    *********************************************/

    /*********************************************
    HSV TO RGB
    *********************************************/
    typedef struct {
        double r;       // percent
        double g;       // percent
        double b;       // percent
    } rgb;

    __device__ rgb hsl2rgb(double h, double sl, double l)
    {
        double v;
        double r,g,b;

        r = l;   // default to gray
        g = l;
        b = l;
        v = (l <= 0.5) ? (l * (1.0 + sl)) : (l + sl - l * sl);
        if (v > 0) {
            double m;
            double sv;
            int sextant;
            double fract, vsf, mid1, mid2;
            m = l + l - v;
            sv = (v - m ) / v;
            h *= 6.0;
            sextant = (int)h;
            fract = h - sextant;
            vsf = v * sv * fract;
            mid1 = m + vsf;
            mid2 = v - vsf;
            switch (sextant) {
                    case 0:
                        r = v;
                        g = mid1;
                        b = m;
                        break;
                    case 1:
                        r = mid2;
                        g = v;
                        b = m;
                        break;
                    case 2:
                        r = m;
                        g = v;
                        b = mid1;
                        break;
                    case 3:
                        r = m;
                        g = mid2;
                        b = v;
                        break;
                    case 4:
                        r = mid1;
                        g = m;
                        b = v;
                        break;
                    case 5:
                        r = v;
                        g = m;
                        b = mid2;
                        break;
            }
        }
        rgb rgb_;
        rgb_.r = (r * 255.0f);
        rgb_.g = (g * 255.0f);
        rgb_.b = (b * 255.0f);
        return rgb_;
    }

    /**************************************************
    *************************************************/

    __device__ static unsigned long next = 1;

    __device__ double my_rand() {
        next = next * 1103515245 + 12345;
        return((double)((unsigned)(next/65536) % 32768))/32768.0;
    }

    __device__ rgb getColor(int n, int MAXCOLORS) {
        double val = (360.0*n)/MAXCOLORS;
        double hue = val/360.0;
        double lightness = 0.6;
        double saturation = 1.0;
        return hsl2rgb(hue, saturation, lightness);
    }

    __device__ rgb getBWColor(int n, int max_n, int MAXCOLORS) {
        rgb rgb_;
        rgb_.r = (n*MAXCOLORS)/max_n;
        rgb_.g = (n*MAXCOLORS)/max_n;
        rgb_.b = (n*MAXCOLORS)/max_n;
        return rgb_;
    }

    __device__ cuDoubleComplex getFuncVal(cuDoubleComplex z, cuDoubleComplex c, int categ) {
        switch(categ) {
            case 1:
                return cuCadd(cuCmul(z,z),c);
            case 2:
                return cuCadd(cuCpow(z,make_cuDoubleComplex(3,0)),c);
            case 3:
                return cuCadd(cuCpow(z,make_cuDoubleComplex(4,0)),c);
            case 4:
                return cuCadd(cuCpow(z,make_cuDoubleComplex(5,0)),c);
            case 5:
                return cuCadd(cuCexp(z),c);
            case 6:
                return cuCadd(cuCexp(cuCpow(z,make_cuDoubleComplex(3,0))),c);
            case 7:
                return cuCadd(cuCmul(z,cuCexp(z)),c);
            case 8:
                return cuCadd(cuCmul(cuCmul(z,z),cuCexp(z)),c);
            case 9:
                return cuCadd(cuCmul(cuCpow(z,make_cuDoubleComplex(3,0)),cuCexp(z)),c);
            case 10:
                return cuCadd(cuCpow(cuCsinh(cuCmul(z,z)),make_cuDoubleComplex(2,0)),c);
            case 11:
                return cuCadd(cuCdiv(cuCadd(make_cuDoubleComplex(1,0),cuCadd(cuCmul(make_cuDoubleComplex(-1,0),cuCmul(z,z)), cuCpow(z,make_cuDoubleComplex(5,0)))),cuCadd(make_cuDoubleComplex(2,0),cuCmul(make_cuDoubleComplex(4,0),z))),c);
            case 12:
                return cuCadd(cuCmul(cuCcos(cuCexp(z)),cuCsin(cuCexp(z))),c);
            case 13:
                return cuCadd(cuCcos(z),c);
            case 14:
                return cuCadd(cuCsin(z),c);
            case 15:
                return cuCadd(cuCmul(cuClog(z),cuCcos(z)),c);
            case 16:
                return cuCadd(cuCdiv(cuCcos(z),(z)),c);
            case 17:
                return cuCadd(cuCdiv(cuClog(z),z),c);
            case 18:
                return cuCadd(cuCmul(cuCsinh(z),z),c);
            case 19:
                return cuCadd(cuCmul(cuCsinh(z),cuCmul(cuCcosh(z),cuCmul(cuCsin(z),cuCcos(z)))),c);
            case 20:
                return cuCadd(cuCexp(cuCexp(z)),c);
            default:
                return cuCadd(cuCmul(z,z),c);
        }
    }

    __global__ void fractalForm(uint8_t *mat, int maxZAbs, int maxN, int minN, int decayN, double iReal,
                                double iImg, int categ, double rMin, double rMax, double iMin, double iMax,
                                int H, int W, int numCols) {
        int idx = threadIdx.x + blockIdx.x*blockDim.x;
        if(idx >= H*W)
            return;
        int i_ = idx/W;
        int j_ = idx%W;
        double re = rMin + (i_*(rMax-rMin))/(1.0*H);
        double im = iMin + (j_*(iMax-iMin))/(1.0*W);
        cuDoubleComplex z = make_cuDoubleComplex(re,im);
        cuDoubleComplex c = make_cuDoubleComplex(iReal, iImg);
        size_t n;
        for(n = maxN; n >= minN && cuCabs(z) < maxZAbs; n-=decayN) {
            z = getFuncVal(z, c, categ);
        }
        rgb col = getColor(n/decayN, numCols);
        mat[3*(j_ + i_*W)]=(uint8_t)(int)col.g;
        mat[3*(j_ + i_*W)+1]=(uint8_t)(int)col.r;
        mat[3*(j_ + i_*W)+2]=(uint8_t)(int)col.b; 
    }
    """

    mod = SourceModule(kernel_code, include_dirs=["F:/codebase/dibo/src/prelims/"])
    fractalForm = mod.get_function("fractalForm")

    args = parse_arguments()
    # H = args.height
    # W = H   # Currently only supports square images
    H = len(all_notes)
    W = args.width
    category = args.category

    h_mat = np.zeros(3*H*W, dtype=np.uint8)
    d_mat = drv.mem_alloc(h_mat.nbytes)
    drv.memcpy_htod(d_mat, h_mat)

    threadsPerBlock = BLOCKSIZE
    numBlocks = int((H*W-1)/threadsPerBlock + 1)

    iReal = (interceptRealMin+interceptRealMax)/2
    iImg = (interceptImgMin+interceptImgMax)/2
    while True:
        fractalForm(d_mat, np.int32(maxZAbs), np.int32(maxN), np.int32(minN),
                    np.int32(decayN), np.float64(iReal), np.float64(iImg),
                    np.int32(category), np.float64(realMin), np.float64(realMin + offset), 
                    np.float64(imgMin), np.float64(imgMin + offset), np.int32(H),
                    np.int32(W), np.int32(numColors),
                    block = (threadsPerBlock,1,1), grid=(numBlocks,1));		
        
        drv.memcpy_dtoh(h_mat, d_mat)
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
            # create and play music
            h_mat_ = h_mat.reshape((H,W,3))
            gs_mat = (.2126 * h_mat_[:,:,0] + .7152 * h_mat_[:,:,0] + .0722 * h_mat_[:,:,0]).astype(np.uint8)
            min_gs = np.min(gs_mat, axis=1).reshape((H,1))
            gs_mat -= min_gs
            max_gs = np.max(gs_mat, axis=1)

            music_ = [[None]*W]*H
            for i in range(H):
                thresh1 = max_gs[i]/3.0
                thresh2 = 2 * thresh1
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
            make_wav_f("fractal_song_par.wav", intensity_list_list)
            print("Took", time.time()-t)