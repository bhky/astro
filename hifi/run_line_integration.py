#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

LINE_TABLE = "/home/byung/HIPE/Data/ObsIDs/Lines_Tables/lines.cwleo.sort.ver.alpha.csv"
OBS_TABLE = "/home/byung/HIPE/Data/ObsIDs/Obs_Tables/Obs-HiFipoint-all-bands_vlsr_2020_2.csv"


def find_lines(
        min_freq: float,
        max_freq: float,
        line_table_path: str,
        delimiter: str = ":"
) -> List[Tuple[str, float]]:
    """
    Return list of (species_str, line_freq0_float).
    """
    lines: List[Tuple[str, float]] = []
    with open(line_table_path, "r") as f:
        for row in f.readlines():
            row = row.strip("\n").split(delimiter)
            species = str(row[0])
            line_freq = float(row[1])
            if min_freq <= line_freq <= max_freq:
                lines.append((species, line_freq))
    return lines


@dataclass
class Observation:
    obs_id: int
    band: str
    object_name: str
    vlsr: float


def make_obs_dict(
        obs_table_path: str,
        delimiter: str = ";"
) -> Dict[int, Observation]:
    """
    Return dict of {obs_id: Observation}.
    """
    obs_ids, bands, obj_names, vlsrs = np.loadtxt(
        obs_table_path, unpack=True, usecols=[0, 1, 2, 13],
        skiprows=1, delimiter=delimiter, dtype=str
    )
    obs_dict: Dict[int, Observation] = {}
    for obs_id, band, obj_name, vlsr in zip(obs_ids, bands, obj_names, vlsrs):
        obs_dict[int(obs_id)] = Observation(
            int(obs_id), band, obj_name, float(vlsr)
        )
    return obs_dict


def load_spectrum_dat(dat_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    freqs, fluxes = np.loadtxt(
        dat_path, unpack=True, usecols=[0, 1], skiprows=2
    )
    rms = float(open(dat_path).readline())
    return freqs, fluxes, rms


def get_trapezoid_areas(freqs: np.ndarray, fluxes: np.ndarray) -> np.ndarray:
    areas: List[float] = []
    for i in range(len(freqs) - 1):
        area = (fluxes[i] + fluxes[i + 1]) * (freqs[i + 1] - freqs[i]) / 2.0
        if np.isnan(area):
            area = 0.0
        areas.append(area)
    return np.array(areas)


def obs_freq_at_vlsr(rest_freq: float, vlsr: float) -> float:
    c = 299792.458  # km/s
    return (1.0 - (vlsr / c)) * rest_freq


def find_line_limits(
        freqs: np.ndarray,
        areas: np.ndarray,
        obs_freq: float
) -> Tuple[float, float]:
    ascending_freq_cumulative_areas = np.cumsum(areas)
    descending_freq_cumulative_ares = np.cumsum(areas[::-1])[::-1]
    obs_idx = int(np.min(np.nonzero(np.less_equal(obs_freq, freqs))))

    asc_cumulated_flux_change = 0
    end_idx = obs_idx
    while asc_cumulated_flux_change >= 0:
        asc_cumulated_flux_change = \
            ascending_freq_cumulative_areas[end_idx + 1] - ascending_freq_cumulative_areas[end_idx]
        end_idx += 1

    des_cumulated_flux_change = 0
    start_idx = obs_idx
    while des_cumulated_flux_change >= 0:
        des_cumulated_flux_change = \
            descending_freq_cumulative_ares[start_idx - 1] - descending_freq_cumulative_ares[start_idx]
        start_idx -= 1
    return start_idx, end_idx


def plot(
        lsb_freqs: np.ndarray,
        lsb_fluxes: np.ndarray,
        areas: np.ndarray,
        obs_freq: float,
        start_freq: float,
        end_freq: float,
        usb_freqs: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 1)

    # Plot spectrum.
    axes[0].step(lsb_freqs[:-1], lsb_fluxes[:-1])
    usb_ax = axes[0].twiny()
    usb_ax.step(usb_freqs[:-1], np.ones_like(usb_freqs[:-1]), c="none")  # Dummy.

    # Plot cumulative (flux) areas.
    cumulative_areas = np.cumsum(areas)
    axes[1].step(lsb_freqs[:-1], cumulative_areas)
    reversed_cumulative_areas = np.cumsum(areas[::-1])[::-1]
    axes[1].step(lsb_freqs[:-1], reversed_cumulative_areas)

    # Plot integration limits.
    # todo: use a loop to plot multiple limits for multiple lines.
    axes[0].plot([obs_freq, obs_freq], [-0.5, 0.5], c="b")
    axes[0].plot([start_freq, start_freq], [-0.5, 0.5], c="r")
    axes[0].plot([end_freq, end_freq], [-0.5, 0.5], c="r")
    axes[1].plot([obs_freq, obs_freq], [-0.05, 0.15], c="b")
    axes[1].plot([start_freq, start_freq], [-0.05, 0.15], c="r")
    axes[1].plot([end_freq, end_freq], [-0.05, 0.15], c="r")

    # Axes settings.
    axes[0].set_xlabel("GHz")
    axes[0].set_ylabel("K")
    axes[0].tick_params(which="both", direction="in")
    axes[0].minorticks_on()
    usb_ax.tick_params(which="both", direction="in")
    usb_ax.minorticks_on()
    usb_ax.invert_xaxis()
    axes[1].set_xlabel("GHz")
    axes[1].set_ylabel("K GHz")
    axes[1].tick_params(which="both", direction="in")
    axes[1].minorticks_on()

    plt.show()


def main() -> None:
    dat_path = "Band_5a/Spectra/1342204741.WBS-LSB.sp1.ave.resampled.dat"
    usb_dat_path = "Band_5a/Spectra/1342204741.WBS-USB.sp1.ave.resampled.dat"
    vlsr = -22.7  # km/s
    # rest_freq = 1151.985  # GHz, in LSB
    rest_freq = 1153.127  # GHz, in LSB
    # rest_freq = 1162.912  # GHz, in USB

    freqs, fluxes, _ = load_spectrum_dat(dat_path)
    usb_freqs, _, _ = load_spectrum_dat(usb_dat_path)
    areas = get_trapezoid_areas(freqs, fluxes)

    obs_freq = obs_freq_at_vlsr(rest_freq, vlsr)
    start_idx, end_idx = find_line_limits(freqs, areas, obs_freq)
    start_freq, end_freq = freqs[start_idx], freqs[end_idx]
    plot(freqs, fluxes, areas, obs_freq, start_freq, end_freq, usb_freqs)


if __name__ == "__main__":
    main()
