#!/usr/bin/env python3
"""
Re-sampling spectra from *.sp1.ave.dat files.
"""

import os
from glob import glob
from typing import Tuple

import numpy as np
from astropy import units as u
from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler

ROOT_DIR = "./"
BAND_DIRS = [
    "Band_1a", "Band_1b", "Band_2a", "Band_2b", "Band_3a", "Band_3b",
    "Band_4a", "Band_4b", "Band_5a", "Band_5b", "Band_6a", "Band_6b",
    "Band_7a", "Band_7b"
]
SPECTRA_DIR = "Spectra/"
CENTRAL_FREQ_DICT = {
    # Central frequency in GHz, see handbook.
    "Band_1a": 520.5,
    "Band_1b": 595.5,
    "Band_2a": 676.0,
    "Band_2b": 758.0,
    "Band_3a": 829.5,
    "Band_3b": 909.5,
    "Band_4a": 1005.0,
    "Band_4b": 1084.25,
    "Band_5a": 1176.1,
    "Band_5b": 1253.5,
    "Band_6a": 1494.0,
    "Band_6b": 1638.0,
    "Band_7a": 1747.5,
    "Band_7b": 1847.5,
}
VELOCITY_RESOLUTION_KMS = 1.0


def _make_spectrum(freqs: np.ndarray, fluxes: np.ndarray) -> Spectrum1D:
    fluxes = np.where(np.isnan(fluxes), 0.0, fluxes)
    return Spectrum1D(spectral_axis=freqs * u.GHz, flux=fluxes * u.K)


def load_spectrum_dat(dat_path: str) -> Tuple[Spectrum1D, float]:
    freqs, fluxes = np.loadtxt(
        dat_path, unpack=True, usecols=[0, 1], skiprows=2
    )
    rms = float(open(dat_path).readline())
    return _make_spectrum(freqs, fluxes), rms


def get_resample_freq_step(
        central_freq: float,
        velocity_resolution_kms: float
) -> float:
    """
    Consider: abs(delta_f) = abs(delta_v) * (f / c)
    Here we choose f to be central_freq for the whole band.
    """
    c = 299792.458  # km/s
    return velocity_resolution_kms * central_freq / c


def resample(
        spectrum: Spectrum1D,
        freq_step: float,
        resampler: FluxConservingResampler,
        rms: float
) -> Tuple[Spectrum1D, float]:
    min_freq = np.min(np.array(spectrum.spectral_axis))
    max_freq = np.max(np.array(spectrum.spectral_axis))
    new_disp_grid = np.arange(min_freq, max_freq, freq_step) * u.GHz
    new_spectrum = resampler(spectrum, new_disp_grid)
    new_rms = rms * np.sqrt(len(new_disp_grid) / len(spectrum.spectral_axis))
    return new_spectrum, new_rms


def load_and_resample(
        dat_path: str,
        central_freq: float,
        resampler: FluxConservingResampler
) -> Tuple[Spectrum1D, float]:
    spectrum, rms = load_spectrum_dat(dat_path)
    freq_step = get_resample_freq_step(
        central_freq, VELOCITY_RESOLUTION_KMS
    )
    new_spectrum, new_rms = resample(
        spectrum, freq_step, resampler, rms
    )
    return new_spectrum, new_rms


def write_spectrum_dat(spectrum: Spectrum1D, rms: float, dat_path: str) -> None:
    freqs = np.array(spectrum.spectral_axis)
    fluxes = np.array(spectrum.flux)
    with open(dat_path, "w") as f:
        print(str(rms), file=f)
        print("----------------", file=f)
        for freq, flux in zip(freqs, fluxes):
            print(str(freq) + "\t" + str(flux), file=f)


def main() -> None:
    resampler = FluxConservingResampler()
    for band_dir in BAND_DIRS:
        central_freq = CENTRAL_FREQ_DICT[band_dir]
        print(f"Working on: {band_dir}; central freq: {central_freq}")
        spectra_dir = os.path.join(ROOT_DIR, band_dir, SPECTRA_DIR)

        for dat_path in glob(os.path.join(spectra_dir, "*.sp1.ave.dat")):
            new_spectrum, new_rms = load_and_resample(
                dat_path, central_freq, resampler
            )
            new_dat_file = dat_path.replace(".dat", f".resampled.dat")
            write_spectrum_dat(new_spectrum, new_rms, new_dat_file)


if __name__ == "__main__":
    main()
