import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import matplotlib.cm as cm

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
