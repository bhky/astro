#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Tuple

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
    Return list of (transition_str, line_rest_freq_float).
    """
    lines: List[Tuple[str, float]] = []
    skip_one_line = True
    with open(line_table_path, "r") as f:
        for row in f.readlines():
            if skip_one_line:
                skip_one_line = False
                continue
            row = row.strip("\n").split(delimiter)
            transition = str(row[0])
            line_freq = float(row[2])
            if min_freq <= line_freq <= max_freq:
                lines.append((transition, line_freq))
    return lines


def is_line(
        freqs: np.ndarray,
        fluxes: np.ndarray,
        obs_freq: float,
        rms: float
) -> bool:
    assert min(freqs) <= obs_freq <= max(freqs)
    idx = int(np.min(np.nonzero(np.less_equal(obs_freq, freqs))))
    return min(fluxes[idx - 5: idx + 5]) >= rms * 3.0


@dataclass
class Observation:
    obs_id: int
    band: str
    object_name: str
    vlsr: float


def get_observations(
        obs_table_path: str,
        delimiter: str = ";"
) -> List[Observation]:
    obs_ids, bands, obj_names, vlsrs = np.loadtxt(
        obs_table_path, unpack=True, usecols=[0, 1, 2, 13],
        skiprows=1, delimiter=delimiter, dtype=str
    )
    observations: List[Observation] = []
    for obs_id, band, obj_name, vlsr in zip(obs_ids, bands, obj_names, vlsrs):
        observations.append(
            Observation(int(obs_id), band, obj_name, float(vlsr))
        )
    return observations


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
        lsb_obs_freqs: List[float],
        lsb_start_freqs: List[float],
        lsb_end_freqs: List[float],
        usb_freqs: np.ndarray,
        usb_obs_freqs: List[float],
        usb_start_freqs: List[float],
        usb_end_freqs: List[float],
) -> None:
    fig, ax = plt.subplots()

    # Plot spectrum.
    ax.step(lsb_freqs, lsb_fluxes)
    usb_ax = ax.twiny()
    usb_ax.step(usb_freqs, np.ones_like(usb_freqs), c="none")  # Dummy.

    # Plot integration limits.
    for obs_freq, start_freq, end_freq in zip(lsb_obs_freqs, lsb_start_freqs, lsb_end_freqs):
        ax.plot([obs_freq, obs_freq], [-0.5, 0.5], c="b")
        ax.plot([start_freq, start_freq], [-0.5, 0.5], c="r")
        ax.plot([end_freq, end_freq], [-0.5, 0.5], c="r")
    for obs_freq, start_freq, end_freq in zip(usb_obs_freqs, usb_start_freqs, usb_end_freqs):
        usb_ax.plot([obs_freq, obs_freq], [-0.5, 0.5], c="b")
        usb_ax.plot([start_freq, start_freq], [-0.5, 0.5], c="r")
        usb_ax.plot([end_freq, end_freq], [-0.5, 0.5], c="r")

    # ax settings.
    ax.set_xlabel("GHz")
    ax.set_ylabel("K")
    ax.set_xlim(min(lsb_freqs), max(lsb_freqs))
    ax.tick_params(which="both", direction="in")
    ax.minorticks_on()
    usb_ax.set_xlim(min(usb_freqs), max(usb_freqs))
    usb_ax.tick_params(which="both", direction="in")
    usb_ax.minorticks_on()
    usb_ax.invert_xaxis()

    plt.show()


def plot_observation(observation: Observation) -> None:
    base_path = f"Band_{observation.band}/Spectra/{observation.obs_id}"
    vlsr = observation.vlsr
    object_name = observation.object_name
    lsb_dat_path = f"{base_path}.WBS-LSB.sp1.ave.resampled.dat"
    usb_dat_path = f"{base_path}.WBS-USB.sp1.ave.resampled.dat"

    lsb_freqs, lsb_fluxes, lsb_rms = load_spectrum_dat(lsb_dat_path)
    usb_freqs, usb_fluxes, usb_rms = load_spectrum_dat(usb_dat_path)
    assert len(lsb_freqs) == len(usb_freqs)

    lsb_areas = get_trapezoid_areas(lsb_freqs, lsb_fluxes)
    usb_areas = get_trapezoid_areas(usb_freqs, usb_fluxes)

    lsb_lines = find_lines(min(lsb_freqs), max(lsb_freqs), LINE_TABLE)
    usb_lines = find_lines(min(usb_freqs), max(usb_freqs), LINE_TABLE)

    # LSB.
    lsb_transitions: List[str] = []
    lsb_obs_freqs: List[float] = []
    lsb_start_freqs: List[float] = []
    lsb_end_freqs: List[float] = []
    for transition, rest_freq in lsb_lines:
        obs_freq = obs_freq_at_vlsr(rest_freq, vlsr)
        if not min(lsb_freqs) <= obs_freq <= max(lsb_freqs):
            continue
        if not is_line(lsb_freqs, lsb_fluxes, obs_freq, lsb_rms):
            continue
        start_idx, end_idx = find_line_limits(lsb_freqs, lsb_areas, obs_freq)
        start_freq, end_freq = lsb_freqs[start_idx], lsb_freqs[end_idx]
        lsb_transitions.append(transition)
        lsb_obs_freqs.append(obs_freq)
        lsb_start_freqs.append(start_freq)
        lsb_end_freqs.append(end_freq)
    # USB.
    usb_transitions: List[str] = []
    usb_obs_freqs: List[float] = []
    usb_start_freqs: List[float] = []
    usb_end_freqs: List[float] = []
    for transition, rest_freq in usb_lines:
        obs_freq = obs_freq_at_vlsr(rest_freq, vlsr)
        if not min(usb_freqs) <= obs_freq <= max(usb_freqs):
            continue
        if not is_line(usb_freqs, usb_fluxes, obs_freq, usb_rms):
            continue
        start_idx, end_idx = find_line_limits(usb_freqs, usb_areas, obs_freq)
        start_freq, end_freq = usb_freqs[start_idx], usb_freqs[end_idx]
        usb_transitions.append(transition)
        usb_obs_freqs.append(obs_freq)
        usb_start_freqs.append(start_freq)
        usb_end_freqs.append(end_freq)

    plot(
        lsb_freqs, lsb_fluxes, lsb_obs_freqs, lsb_start_freqs, lsb_end_freqs,
        usb_freqs, usb_obs_freqs, usb_start_freqs, usb_end_freqs
    )


def main() -> None:
    # dat_path = "Band_5a/Spectra/1342204741.WBS-LSB.sp1.ave.resampled.dat"
    # usb_dat_path = "Band_5a/Spectra/1342204741.WBS-USB.sp1.ave.resampled.dat"
    # vlsr = -22.7  # km/s
    # rest_freq = 1151.985  # GHz, in LSB
    # rest_freq = 1153.127  # GHz, in LSB
    # rest_freq = 1162.912  # GHz, in USB

    observation = Observation(1342204741, "5a", "AFGL 5379", -22.7)
    plot_observation(observation)


if __name__ == "__main__":
    main()
