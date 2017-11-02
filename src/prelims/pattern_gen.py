import numpy as np
import cv2

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
    if args.which == 1:
        hits = np.zeros((args.height, args.width))
    elif args.which == 2:
        hits = np.zeros((args.patch_height, args.patch_width))
    elif args.which == 3:
        hits = np.zeros((args.height, args.width))
    max_hits = 1
    x_hit = x_init
    y_hit = y_init

    rand_hits = np.arange(256*256*256)
    np.random.shuffle(rand_hits)
    print("shuffled.")

    for it in range(args.n_iter):
        if args.which == 1:
            (x_hit, y_hit) = sym_icon_f(x_hit,y_hit,args)
        elif args.which == 2:
            (x_hit, y_hit) = quilt_f(x_hit,y_hit,args)
        elif args.which == 3:
            (x_hit, y_hit) = fractal_f(x_hit,y_hit,args)
        (xp,yp) = hit_pixel(x_hit, y_hit, args)
        if (args.which == 1 and xp < args.width and yp < args.height)\
             or (args.which == 2 and xp < args.patch_width and yp < args.patch_height)\
             or (args.which == 3 and xp < args.width and yp < args.height):
            hits[yp,xp] += 1
            if hits[yp,xp] > max_hits:
                max_hits = hits[yp,xp]
        if not args.parallel_hits_map and args.iter_freq_show and it%args.iter_freq_show == 0:
            col_img_to_show = np.zeros((hits.shape[0], hits.shape[1], 3), dtype=np.uint8)
            hits_ = hits.copy().astype(np.int)
            #print(np.max(hits_))
            hits_ = rand_hits[hits_]
            col_img_to_show[:,:,0] = hits_%256
            col_img_to_show[:,:,1] = ((hits_-col_img_to_show[:,:,0])/256)%256
            col_img_to_show[:,:,2] = (hits_ - col_img_to_show[:,:,1]*256 - col_img_to_show[:,:,0])/(256*256)
            cv2.imshow("dynamic_fractal", col_img_to_show)
            cv2.waitKey(1)
    return (hits, max_hits)