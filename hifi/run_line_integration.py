#!/usr/bin/env python3

import os
from dataclasses import dataclass
from typing import List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

LINE_TABLE_PATH = "/home/byung/HIPE/Data/ObsIDs/Lines_Tables/lines.cwleo.sort.ver.alpha.edited.csv"
SUPP_LINE_TABLE_PATH = "/home/byung/HIPE//Data/ObsIDs/Lines_Tables/linie_hifi_band17_n_oproczIRC.edited.csv"

OBS_TABLE_PATH = "/home/byung/HIPE/Data/ObsIDs/Obs_Tables/Obs-HiFipoint-all-bands_vlsr_2020_2.csv"


def find_lines(
        min_freq: float,
        max_freq: float,
        line_table_path: str = LINE_TABLE_PATH,
        supp_line_table_path: str = SUPP_LINE_TABLE_PATH,
        delimiter: str = ";"
) -> List[Tuple[str, str, float]]:
    name_set: Set[str] = set()
    quantum_number_set: Set[str] = set()
    lines: List[Tuple[str, str, float]] = []
    with open(line_table_path, "r") as f:
        rows = f.readlines()[1:]
        for row_str in rows:
            row = row_str.strip("\n").split(delimiter)
            name = str(row[0]).strip()
            line_freq = float(row[2])
            quantum_number = str(row[6])
            if min_freq <= line_freq <= max_freq:
                lines.append((name, quantum_number, line_freq))
                name_set.add(name)
                quantum_number_set.add(quantum_number)
    with open(supp_line_table_path, "r") as f:
        rows = f.readlines()[1:]
        for row_str in rows:
            row = row_str.strip("\n").split(delimiter)
            name = str(row[0]).strip()
            quantum_number = str(row[1])
            line_freq = float(row[2])
            if min_freq <= line_freq <= max_freq:
                if name in name_set and quantum_number in quantum_number_set:
                    continue
                lines.append((name, quantum_number, line_freq))
    return lines


def is_line(
        freqs: np.ndarray,
        fluxes: np.ndarray,
        obs_freq: float,
        rms: float
) -> bool:
    assert min(freqs) <= obs_freq <= max(freqs)
    idx = int(np.min(np.nonzero(np.less_equal(obs_freq, freqs))))
    # Assumption: should have flux larger than, e.g., 3 * rms, for 10 channels.
    return min(fluxes[idx - 5: idx + 5]) >= rms * 3.0


@dataclass
class Observation:
    obs_id: int
    band: str
    object_name: str
    vlsr: float


def get_observations(
        obs_table_path: str = OBS_TABLE_PATH,
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


def obs_freq_at_vlsr(rest_freq: float, vlsr: float) -> float:
    c = 299792.458  # km/s
    return (1.0 - (vlsr / c)) * rest_freq


def find_line_limits(
        freqs: np.ndarray,
        fluxes: np.ndarray,
        rms: float,
        obs_freq: float
) -> Tuple[float, float]:
    obs_idx = int(np.min(np.nonzero(np.less_equal(obs_freq, freqs))))
    end_idx = obs_idx
    for asc_flux in fluxes[obs_idx:]:
        if asc_flux <= rms:
            break
        end_idx += 1
    start_idx = obs_idx
    for des_flux in fluxes[:obs_idx + 1][::-1]:
        if des_flux <= rms:
            break
        start_idx -= 1
    # Fix possible out of bound cases, e.g., when line is at boundary.
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(freqs) - 1)
    return start_idx, end_idx


def make_plot(
        observation: Observation,
        lsb_rms: float,
        lsb_freqs: np.ndarray,
        lsb_fluxes: np.ndarray,
        lsb_obs_freqs: List[float],
        lsb_start_freqs: List[float],
        lsb_end_freqs: List[float],
        lsb_transitions: List[str],
        usb_freqs: np.ndarray,
        usb_obs_freqs: List[float],
        usb_start_freqs: List[float],
        usb_end_freqs: List[float],
        usb_transitions: List[str],
        show_only: bool = True
) -> None:
    fig, ax = plt.subplots()

    # Plot spectrum.
    ax.step(lsb_freqs, lsb_fluxes, lw=1)
    usb_ax = ax.twiny()
    usb_ax.step(usb_freqs, np.ones_like(usb_freqs), c="none")  # Dummy.

    lsb_obs_freq_mark_range = [-7 * lsb_rms, 6 * lsb_rms]
    lsb_start_freq_mark_range = [-5 * lsb_rms, 5 * lsb_rms]
    lsb_end_freq_mark_range = [-5 * lsb_rms, 5 * lsb_rms]
    usb_obs_freq_mark_range = [-7 * lsb_rms, 20 * lsb_rms]
    usb_start_freq_mark_range = [-5 * lsb_rms, 5 * lsb_rms]
    usb_end_freq_mark_range = [-5 * lsb_rms, 5 * lsb_rms]
    if show_only:
        font_size = 8
        y_lim = ()
    else:
        font_size = 5
        y_lim = (-7 * lsb_rms, 25 * lsb_rms)

    # Plot integration limits.
    for obs_freq, start_freq, end_freq, transition in \
            zip(lsb_obs_freqs, lsb_start_freqs, lsb_end_freqs, lsb_transitions):
        ax.plot([obs_freq, obs_freq], lsb_obs_freq_mark_range, c="blue", lw=1)
        ax.plot([start_freq, start_freq], lsb_start_freq_mark_range, c="red", lw=1)
        ax.plot([end_freq, end_freq], lsb_end_freq_mark_range, c="red", lw=1)
        # Note: when "position" is used, the first two coordinates are dummy.
        ax.text(
            0, 0, transition,
            position=(obs_freq, 11 * lsb_rms), ha="center", va="bottom",
            fontsize=font_size, color="blue", rotation=90
        )
    for obs_freq, start_freq, end_freq, transition in \
            zip(usb_obs_freqs, usb_start_freqs, usb_end_freqs, usb_transitions):
        usb_ax.plot([obs_freq, obs_freq], usb_obs_freq_mark_range, c="cyan", lw=1)
        usb_ax.plot([start_freq, start_freq], usb_start_freq_mark_range, c="red", lw=1)
        usb_ax.plot([end_freq, end_freq], usb_end_freq_mark_range, c="red", lw=1)
        # Note: when "position" is used, the first two coordinates are dummy.
        usb_ax.text(
            0, 0, transition,
            position=(obs_freq, 25 * lsb_rms), ha="center", va="bottom",
            fontsize=font_size, color="cyan", rotation=90
        )

    # axes settings.
    ax.set_xlabel("GHz")
    ax.set_ylabel("K")
    ax.set_xlim(min(lsb_freqs), max(lsb_freqs))
    if not show_only:
        ax.set_ylim(*y_lim)
    ax.tick_params(which="both", direction="in")
    ax.minorticks_on()
    usb_ax.set_xlim(min(usb_freqs), max(usb_freqs))
    usb_ax.tick_params(which="both", direction="in")
    usb_ax.minorticks_on()
    usb_ax.invert_xaxis()

    ax.set_title(
        f"{observation.obs_id}.WBS, "
        f"{observation.band}, "
        f"{observation.object_name}, "
        f"v_lsr = {observation.vlsr}"
    )

    if show_only:
        plt.show()
    else:
        file_path = f"./Figures/CW_Leo/cwleo.{observation.band}.{observation.obs_id}.WBS.sp1.ave.resampled.lines.pdf"
        plt.savefig(file_path)
        plt.close()


def plot_observation(observation: Observation) -> None:
    base_path = f"Band_{observation.band}/Spectra/{observation.obs_id}"
    vlsr = observation.vlsr
    lsb_dat_path = f"{base_path}.WBS-LSB.sp1.ave.resampled.dat"
    usb_dat_path = f"{base_path}.WBS-USB.sp1.ave.resampled.dat"

    if not os.path.exists(lsb_dat_path) or not os.path.exists(usb_dat_path):
        return

    lsb_freqs, lsb_fluxes, lsb_rms = load_spectrum_dat(lsb_dat_path)
    usb_freqs, usb_fluxes, usb_rms = load_spectrum_dat(usb_dat_path)

    lsb_lines = find_lines(min(lsb_freqs), max(lsb_freqs))
    usb_lines = find_lines(min(usb_freqs), max(usb_freqs))

    # LSB.
    lsb_transitions: List[str] = []
    lsb_obs_freqs: List[float] = []
    lsb_start_freqs: List[float] = []
    lsb_end_freqs: List[float] = []
    for name, quantum_number, rest_freq in lsb_lines:
        obs_freq = obs_freq_at_vlsr(rest_freq, vlsr)
        if not min(lsb_freqs) <= obs_freq <= max(lsb_freqs):
            continue
        if not is_line(lsb_freqs, lsb_fluxes, obs_freq, lsb_rms):
            continue
        start_idx, end_idx = find_line_limits(lsb_freqs, lsb_fluxes, lsb_rms, obs_freq)
        start_freq, end_freq = lsb_freqs[start_idx], lsb_freqs[end_idx]
        lsb_transitions.append(name + " " + quantum_number)
        lsb_obs_freqs.append(obs_freq)
        lsb_start_freqs.append(start_freq)
        lsb_end_freqs.append(end_freq)
    # USB.
    usb_transitions: List[str] = []
    usb_obs_freqs: List[float] = []
    usb_start_freqs: List[float] = []
    usb_end_freqs: List[float] = []
    for name, quantum_number, rest_freq in usb_lines:
        obs_freq = obs_freq_at_vlsr(rest_freq, vlsr)
        if not min(usb_freqs) <= obs_freq <= max(usb_freqs):
            continue
        if not is_line(usb_freqs, usb_fluxes, obs_freq, usb_rms):
            continue
        start_idx, end_idx = find_line_limits(usb_freqs, usb_fluxes, usb_rms, obs_freq)
        start_freq, end_freq = usb_freqs[start_idx], usb_freqs[end_idx]
        usb_transitions.append(name + " " + quantum_number)
        usb_obs_freqs.append(obs_freq)
        usb_start_freqs.append(start_freq)
        usb_end_freqs.append(end_freq)

    make_plot(
        observation, lsb_rms, lsb_freqs, lsb_fluxes,
        lsb_obs_freqs, lsb_start_freqs, lsb_end_freqs, lsb_transitions,
        usb_freqs, usb_obs_freqs, usb_start_freqs, usb_end_freqs, usb_transitions
    )


def main() -> None:
    observations = get_observations()
    for observation in observations:
        if observation.object_name == "IRC+10216":
            plot_observation(observation)


if __name__ == "__main__":
    main()
