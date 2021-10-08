"""
usage:
./script.py a.XX-LSB.sp1.ave.resampled.dat a.XX-USB.sp1.ave.resampled.dat \
            b.XX-LSB.sp1.ave.resampled.dat b.XX-USB.sp1.ave.resampled.dat ...
Note that LSB must come first for each pair of spectra.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

# Import Bartek's function
from lines_seeker import lines_search

# Bartek's tables
line_table = '/home/byung/HIPE/Data/ObsIDs/Lines_Tables/result4_2.csv'
obs_table = '/home/byung/HIPE/Data/ObsIDs/Obs_Tables/Obs-HiFipoint-all-bands_ver3_4.csv'

# Make dicts ObsID:object and ObsID:vlsr
ObsIDs, objs, vlsrs = np.loadtxt(obs_table, unpack=True, usecols=[0, 2, 13],
                                 skiprows=4, delimiter=';', dtype=str)
obj_dict = dict(zip(ObsIDs, objs))
vlsr_dict = dict(zip(ObsIDs, vlsrs.astype(float)))

for i in range(1, len(sys.argv) - 1, 2):

    # Prepare figure
    fig, ax1 = plt.subplots()

    # Read each ObsID from file name
    ObsID = sys.argv[i].split('.')[0]

    # Read rms
    rms = float(open(sys.argv[i]).readline())

    # Read freq and flux
    lsb_freq_array, lsb_flux_array = np.loadtxt(sys.argv[i], unpack=True,
                                                usecols=[0, 1], skiprows=2)
    usb_freq_array, usb_flux_array = np.loadtxt(sys.argv[i + 1], unpack=True,
                                                usecols=[0, 1], skiprows=2)

    # Plot
    ax1.step(lsb_freq_array, lsb_flux_array, c='black', lw=0.3)
    ax1.plot([min(lsb_freq_array), max(lsb_freq_array)], [rms * 3, rms * 3],
             c='purple', lw=0.2)
    # USB plot is dummy
    ax2 = ax1.twiny()
    ax2.step(usb_freq_array, usb_flux_array, c='none')

    # Label the possible lines
    c = 299792.458  # km/s
    # For LSB
    lsb_line_list = lines_search(lsb_freq_array[0], lsb_freq_array[-1],
                                 line_table)
    if len(lsb_line_list) != 0:
        for line_molecule, line_trans, line_freq0 in lsb_line_list:
            line_freq = (1 - (vlsr_dict[ObsID] / c)) * line_freq0
            ax1.plot([line_freq, line_freq], [-7 * rms, 6 * rms], c='red')
            # Note:, when "position" is used, the first two coor args are dummy
            ax1.text(0, 0, line_molecule,
                     position=(line_freq, 7 * rms), ha='center', va='bottom',
                     fontsize=8, color='red', rotation=90)

    # For USB
    usb_line_list = lines_search(usb_freq_array[0], usb_freq_array[-1],
                                 line_table)
    if len(usb_line_list) != 0:
        for line_molecule, line_trans, line_freq0 in usb_line_list:
            line_freq = (1 - (vlsr_dict[ObsID] / c)) * line_freq0
            ax2.plot([line_freq, line_freq], [13 * rms, 20 * rms], c='blue')
            # Note:, when "position" is used, the first two coor args are dummy
            ax2.text(0, 0, line_molecule,
                     position=(line_freq, 12 * rms), ha='center', va='top',
                     fontsize=8, color='blue', rotation=90)

    # Set axes parameters
    ax1.set_ylim(-7 * rms, 20 * rms)
    ax1.set_ylabel("Antenna Temperature (K)")
    backend = sys.argv[i].split('.')[1][:3]

    ax1.set_xlabel("LSB Frequency (GHz)\n" +
                   "rms = " + str(rms), fontsize=11)

    ax1.set_xlim(min(lsb_freq_array), max(lsb_freq_array))
    ax1.tick_params(which="both", direction="in")
    ax1.minorticks_on()

    ax2.set_xlabel(ObsID + "." + backend + ", " + obj_dict[ObsID] + ", Vlsr = "
                   + str(vlsr_dict[ObsID]) + "\n" + "USB Frequency (GHz)",
                   fontsize=11)
    ax2.set_xlim(min(usb_freq_array), max(usb_freq_array))
    ax2.tick_params(which="both", direction="in")
    ax2.minorticks_on()
    ax2.invert_xaxis()

    plt.tight_layout()

    # Save the fig
    fig.savefig(ObsID + "." + backend + ".sp1.ave.resampled.pdf")
    plt.close()
