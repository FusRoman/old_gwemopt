
import os
import glob
import optparse

import numpy as np
from astropy.time import Time

import matplotlib
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 24}
matplotlib.rc('xtick', labelsize=22)
matplotlib.rc('ytick', labelsize=22)
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

outputDir = '../plots/telescopes/'
if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

color1 = 'cornflowerblue'

filename = "../data/S200213t/telescopes.dat"

teles = [line.rstrip('\n') for line in open(filename,'r')]

diffs = [[0.8,0.92], [0.95,0.95],
         [0.80,1.03], [1.15,1.0],
         [0.9,1.05], [0.95, 0.95],
         [0.95, 0.92]]

plotName = os.path.join(outputDir,'S200213t_performance.pdf')
fig = plt.figure(figsize=(8,6))
ax = plt.subplot(111)

absmag = -16.0
event_mu, event_std = 224, 90
event_mag = 5.0*(np.log10(event_mu*1e6)-1.0) + absmag
event_mag_lower = 5.0*(np.log10((event_mu-event_std)*1e6)-1.0) + absmag
event_mag_upper = 5.0*(np.log10((event_mu+event_std)*1e6)-1.0) + absmag
event_mag_diff = event_mag_upper-event_mag_lower

plt.plot([-5,100], [event_mag,event_mag], alpha=1.0, color='g')
rect1 = Rectangle((-5, event_mag_lower), 105, event_mag_diff, alpha=0.3, color='g')
ax.add_patch(rect1)

plt.xlabel('Probability Enclosed')
plt.ylabel('Limiting Magnitude')
for tele, diff in zip(teles,diffs):
    telesplit = tele.split(" ")
    tel, limmag, coverage = telesplit[0], float(telesplit[1]), float(telesplit[2])
    #plt.text(model[3], model[1], model[0])
    plt.plot(coverage, limmag, marker='o', markerfacecolor=color1,
                markeredgecolor='black', markersize=25)

    diffx, diffy = diff
    x, y = coverage, limmag

    x1text = x*diffx*1.01
    y1text = y*diffy*1.01

    x1arrow = x*diffx*1.01
    y1arrow = y*diffy*1.01

    plt.text(x1text, y1text, tel)
    plt.annotate('', xy=(x,y), xytext=(x1arrow,y1arrow),
                 arrowprops=dict(facecolor='black', arrowstyle='->'),
                 )

plt.xlim([-5,100])
plt.ylim([15.5,22.5])
#plt.gca().invert_yaxis()
plt.savefig(plotName,bbox_inches='tight')
plt.close()


