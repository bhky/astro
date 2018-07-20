"""
usage:
./script.py a.sp0.dat a.sp1.dat b.sp0.dat b.sp1.dat ...
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

## Import Bartek's function
from lines_seeker import lines_search

## Bartek's line table
line_table = '/home/byung/HIPE/Data/ObsIDs/Lines_Tables/result4_2.csv'


for i in range(1, len(sys.argv)-1, 2):

   ## Plot the two spectra, above is before reduction, below is after
   fig, [ax0, ax1] = plt.subplots(2)

   order_rms_list = open(sys.argv[i+1]).readline().split()
   ## Use average rms for plotting
   rms_list = []
   for r in range(1, len(order_rms_list, 2)):
      rms_list.append(order_rms_list[r])
   rms = np.mean(rms_list)

   ## Read and plot freq and flux
   freq0_array, flux0_array = np.loadtxt(sys.argv[i], unpack=True, \
                                         usecols=[0,1])
   freq1_array, flux1_array = np.loadtxt(sys.argv[i+1], unpack=True, \
                                         usecols=[0,1], skiprows=2)

   ax0.plot(freq0_array, flux0_array)
   ax1.plot(freq1_array, flux1_array)


   ## Label the possible lines
   line_list = lines_search(freq1_array[0], freq1_array[-1], line_table)
   if len(line_list) != 0:
      for line_molecule, line_trans, line_freq in line_list:
         ax1.plot([line_freq, line_freq], [0, 8*rms], 'red')
         ## Note:, when "position" is used, the first two coor args are dummy
         ax1.text(0, 0, line_molecule+'('+line_trans+')', \
                  position=(line_freq, 10*rms), ha='center', va='bottom', \
                  fontsize=8, color='red', rotation=90)


   ## Set axes parameters
   for ax in [ax0, ax1]:
      ax.set_xlabel("Frequency (GHz)")
      ax.set_ylabel("Antenna Temperature (K)")
      ax.set_xlim(min(freq0_array), max(freq0_array))
      ax.tick_params(which="both", direction="in")
      ax.minorticks_on()

   ax1.set_ylim(-7*rms, 20*rms)

   ax0.set_title(sys.argv[i][:-8])

#   order_rms = ''
#   for j in range(0, len(order_rms_list), 2):
#      order_rms = order_rms+"Subband "+str(j+1)+\
#      ": order="+order_rms_list[j]+", rms="+\
#      "%.5f" % float(order_rms_list[j+1]) + " "
#   ax1.set_title(order_rms, fontsize=8)

   order_list = []
   for j in range(0, len(order_rms_list), 2):
      order_list.append(float(order_rms_list[j]))
   rms_list = []
   for j in range(0, len(order_rms_list), 2):
      rms_list.append(float(order_rms_list[j+1][:7]))
   ax1.set_title("order = "+str(list(reversed(order_list)))+\
                 ", rms = "+str(list(reversed(rms_list))), fontsize=8)   

   plt.tight_layout()


   ## Save the fig
   fig.savefig(sys.argv[i][:-8]+".pdf")
   plt.close()

