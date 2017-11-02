import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description = 'description',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-height', '--height', type=int,
                        help="Height of image",
                        required=True)
    parser.add_argument('-width', '--width', type=int,
                        help="Width of image.",
                        required=True)
    parser.add_argument('-patch_height', '--patch_height', type=int,
                        help="Height of a patch",
                        default=30)
    parser.add_argument('-patch_width', '--patch_width', type=int,
                        help="Width of a patch.",
                        default=30)
    parser.add_argument('-category', '--category', type=int,
                        help="Category of set.",
                        default=1)
    parser.add_argument('-which', '--which', type=int,
                        help="icon[1]/quit[2]",
                        default=1)
    parser.add_argument('-map_type', '--map_type', type=int,
                        help="0/1",
                        default=0)
    parser.add_argument('-num_proc', '--num_proc', type=int,
                        help="num of processes",
                        default=16)
    parser.add_argument('-iter_freq_show', '--iter_freq_show', type=int,
                        help="iterations image show frequency",
                        default=0)
    parser.add_argument('-parallel_hits_map', '--parallel_hits_map', type=int,
                        help="To compute hits map in parallel or sequential",
                        default=1)
    parser.add_argument('-ffmpeg_exe_path', '--ffmpeg_exe_path', type=str,
                        help="Category of set.",
                        default="G:/packages/ffmpeg/bin/ffmpeg.exe")
    parser.add_argument('-n_iter', '--n_iter', type=int,
                        help="No. of iterations.",
                        default=500000)
    parser.add_argument('-scale_height', '--scale_height', type=float,
                        help="scaleH.",
                        default=0.37)
    parser.add_argument('-scale_width', '--scale_width', type=float,
                        help="scaleW.",
                        default=0.37)
    parser.add_argument('-x_init', '--x_init', type=float,
                        help="x_init.",
                        default=0.001)
    parser.add_argument('-y_init', '--y_init', type=float,
                        help="y_init.",
                        default=0.002)
    parser.add_argument('-cluster_prop', '--cluster_prop', type=float,
                        help="y_init.",
                        default=0.1)
    return parser

def parse_arguments(raw=None):
    args = argument_parser().parse_args(raw)
    return args