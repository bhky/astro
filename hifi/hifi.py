#!/usr/bin/env python2.7
"""
The HIFISources class has method to find which sources have observations
covering the input list of frequencies.
----------
2018-07-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import constants
from scipy.optimize import curve_fit


class HIFISources(object):
    """
    Example:
    hifi_obs = hifi.HIFISources(<obs_table>)
    """
    def __init__(self, obs_table, spectra_dir, sep=';', header=0, 
                 index_col=['obsid'], skiprows=[1, 2, 3], *args, **kwargs):
        self.data = pd.read_table(obs_table, sep=sep, header=header,
                                  index_col=index_col, skiprows=skiprows,
                                  *args, **kwargs)
        self.all_objs = sorted(list(self.get_obs_contain_freq()))
        self.spectra_dir = spectra_dir

    def get_obs_contain_freq(self, freq=None):
        """
        Example:
        hifi_obs.get_obs_contain_freq(freq=<freq>)
        """
        if freq is None:
            return set(self.data['object'])
        out_dict = {}
        # condition = (
        #     (
        #         (self.data['obsFreqUsbMin'] < freq) &
        #         (self.data['obsFreqUsbMax'] > freq)
        #     ) | (
        #         (self.data['obsFreqLsbMin'] < freq) &
        #         (self.data['obsFreqLsbMax'] > freq)
        #     )
        # )
        condition_lsb = (
            (self.data['obsFreqLsbMin'] < freq) &
            (self.data['obsFreqLsbMax'] > freq)
        )
        condition_usb = (
            (self.data['obsFreqUsbMin'] < freq) &
            (self.data['obsFreqUsbMax'] > freq)
        )
        objs_series_list = [
            self.data['object'][condition_lsb],
            self.data['object'][condition_usb],
        ]
        bands_list = ['LSB', 'USB']
        for objs_series, band in zip(objs_series_list, bands_list):
            for idx in objs_series.index:
                if objs_series[idx] not in out_dict:
                    out_dict[objs_series[idx]] = [[idx, band]]
                else:
                    out_dict[objs_series[idx]].append([idx, band])
        return out_dict

    def plot_freq_ranges(self, ax=None):
        """
        Example:
        hifi_obs.plot_freq_ranges()
        """
        if ax is None:
            fig, ax = plt.subplots()
        y = 0
        for obj in self.all_objs:
            freqs_df = self.data[self.data['object'] == obj][
                ['obsFreqUsbMin', 'obsFreqUsbMax',
                 'obsFreqLsbMin', 'obsFreqLsbMax']
            ]
            usb_pairs_list = zip(freqs_df['obsFreqUsbMin'],
                                 freqs_df['obsFreqUsbMax'])
            lsb_pairs_list = zip(freqs_df['obsFreqLsbMin'],
                                 freqs_df['obsFreqLsbMax'])
            usb_lsb_pairs_list = zip(usb_pairs_list, lsb_pairs_list)
            for (x1, x2), (x3, x4) in usb_lsb_pairs_list:
                ax.plot([x1, x2], [y, y], linewidth=2, color='r')
                ax.plot([x3, x4], [y, y], linewidth=2, color='b')
            y += 1
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Object Index')
        ax.set_title('Frequency Coverages')

    def get_vlsr(self, freq=None, fitting_coverage=0.8, peak_to_rms=5, 
                 gaussian_sigma0=0.1):
        """
        Example:
        hifi_obs.get_vlsr(freq=<freq>)
        """
        if freq is None:
            vlsr_dict = zip(self.all_objs, np.zeros(len(self.all_objs)))
            return vlsr_dict
        obs_dict = self.get_obs_contain_freq(freq=freq)
        params_dict = {}
        for obj in obs_dict.keys():
            # Find the averaged vlsr if an object has more than one obsid
            # covering the given line.
            vlsr_list = []
            for obsid, band in obs_dict[obj]:
                fits_name = self.spectra_dir + str(obsid) + '.WBS-' + \
                            band + '.sp1.ave.fits'
                try:
                    sp = fits.open(fits_name)
                except IOError:
                    continue
                freq_array = sp[1].data['wave']
                flux_array = sp[1].data['flux']
                sp.close()

                idx = np.argwhere(
                    (freq_array > freq - (fitting_coverage * 0.5)) &
                    (freq_array < freq + (fitting_coverage * 0.5))
                ).ravel()

                # Use the region next to the line for estimating rms.
                rms_idx = np.argwhere(
                    (freq_array > freq + (fitting_coverage * 0.5)) &
                    (freq_array < freq + 2 * (fitting_coverage * 0.5))
                ).ravel()
                rms = self._rms(flux_array[rms_idx])

                peak_flux = np.max(flux_array[idx])
                peak_idx = np.argmax(flux_array[idx])
                if peak_flux / rms >= peak_to_rms:
                    popt, pcov = curve_fit(
                        self._gaussian,
                        freq_array[idx], flux_array[idx],
                        p0=[peak_flux, freq_array[idx][peak_idx], 
                            gaussian_sigma0] 
                    )
                    freq_obs = popt[1]
                    vlsr_list.append(
                        self._vlsr(freq, freq_obs) / len(obs_dict[obj])
                    )
            if vlsr_list:
                # Calculate average.
                vlsr = np.mean(vlsr_list)
                params_dict[obj] = [freq_obs, vlsr]
        return params_dict

    def _vlsr(self, freq_0, freq_obs):
        return (-(freq_obs - freq_0) * (constants.c / 1000)) / freq_0

    def _rms(self, data_array):
        return np.sqrt(np.mean(data_array ** 2))

    def _gaussian(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2.0 * sigma ** 2))



