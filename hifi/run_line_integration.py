#!/usr/bin/env python3

import csv
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LINE_TABLE_PATH = "/home/byung/HIPE/Data/ObsIDs/Lines_Tables/lines.objects.ver2.edited.csv"
OBS_TABLE_PATH = "/home/byung/HIPE/Data/ObsIDs/Obs_Tables/Obs-HiFipoint-all-bands_vlsr_2020_2.csv"

VERSION = "4"
OUTPUT_TABLE_PATH = f"/home/byung/HIPE/Data/ObsIDs/cwleo.result.v{VERSION}.csv"


def find_line_identifications(
        min_freq: float,
        max_freq: float,
        object_name: Optional[str],
        line_table_path: str = LINE_TABLE_PATH,
        delimiter: str = ";"
) -> List[Tuple[str, str, float]]:
    df = pd.read_csv(line_table_path, delimiter=delimiter)
    cond_0 = ~df["Species"].str.contains("#")
    cond_1 = min_freq <= df["Freq-GHz(rest frame,redshifted)"]
    cond_2 = df["Freq-GHz(rest frame,redshifted)"] <= max_freq
    if object_name is not None:
        cond_3 = df[object_name] == 1
    else:
        cond_3 = [True] * len(df)
    df_found = df[cond_0 & cond_1 & cond_2 & cond_3]

    names = df_found["Species"].tolist()
    quantum_numbers = df_found["Resolved QNs"].tolist()
    line_freqs = df_found["Freq-GHz(rest frame,redshifted)"].tolist()
    line_identifications = list(zip(names, quantum_numbers, line_freqs))
    return line_identifications


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
    object: str
    vlsr_kms: float


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
        obs_freq: float,
        rms_factor: float = 1.0
) -> Tuple[int, int]:
    obs_idx = int(np.min(np.nonzero(np.less_equal(obs_freq, freqs))))
    end_idx = obs_idx
    for asc_flux in fluxes[obs_idx:]:
        if asc_flux <= rms * rms_factor:
            break
        end_idx += 1
    start_idx = obs_idx
    for des_flux in fluxes[:obs_idx + 1][::-1]:
        if des_flux <= rms * rms_factor:
            break
        start_idx -= 1
    # Fix possible out of bound cases, e.g., when line is at boundary.
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, len(freqs) - 1)
    return start_idx, end_idx


def find_overlapping_ranges(
        ranges: List[Tuple[int, int]]
) -> Dict[int, List[int]]:
    """
    Return dict: {index: indices of overlapping ranges}
    """
    result_dict: Dict[int, List[int]] = {}
    for i in range(len(ranges)):
        start_i, end_i = ranges[i][0], ranges[i][1]
        result_dict[i]: List[int] = []
        for j in range(len(ranges)):
            if j == i:
                continue
            # Check intersections.
            start_j, end_j = ranges[j][0], ranges[j][1]
            cond_1 = start_i <= start_j <= end_i or start_i <= end_j <= end_i
            cond_2 = start_j <= start_i <= end_j or start_j <= end_i <= end_j
            if cond_1 or cond_2:
                result_dict[i].append(j)
    return result_dict


def get_integrated_flux(
        freqs: np.ndarray,
        fluxes: np.ndarray,
) -> float:
    areas: List[float] = []
    for i in range(len(freqs) - 1):
        area = (fluxes[i] + fluxes[i + 1]) * (freqs[i + 1] - freqs[i]) / 2
        areas.append(area)
    return float(sum(areas))


@dataclass
class Line:
    obs_id: int
    band: str
    object: str
    vlsr_kms: float
    side_band: str
    species_name: str
    quantum_number: str
    rest_freq_GHz: float
    observed_freq_GHz: float
    start_freq_GHz: float
    end_freq_GHz: float
    peak_K: float
    integrated_K_GHz: float
    blended_transitions: str = ""


def extract_line_data(
        observation: Observation,
        freqs: np.ndarray,
        fluxes: np.ndarray,
        rms: float,
        line_identifications: List[Tuple[str, str, float]],
        is_lsb: bool
) -> Tuple[List[Line], List[Tuple[int, int]]]:
    side_band = "LSB" if is_lsb else "USB"
    lines: List[Line] = []
    idx_ranges: List[Tuple[int, int]] = []
    for name, quantum_number, rest_freq in line_identifications:
        obs_freq = obs_freq_at_vlsr(rest_freq, observation.vlsr_kms)
        if not min(freqs) <= obs_freq <= max(freqs):
            continue
        if not is_line(freqs, fluxes, obs_freq, rms):
            continue
        start_idx, end_idx = find_line_limits(freqs, fluxes, rms, obs_freq)
        start_freq, end_freq = freqs[start_idx], freqs[end_idx]

        peak_flux = round(max(fluxes[start_idx: end_idx]), 3)
        integrated_flux = get_integrated_flux(
            freqs[start_idx: end_idx], fluxes[start_idx: end_idx]
        )

        lines.append(
            Line(
                observation.obs_id, observation.band,
                observation.object, observation.vlsr_kms,
                side_band, name, quantum_number, rest_freq, obs_freq,
                start_freq, end_freq, peak_flux, integrated_flux
            )
        )

        if not is_lsb:
            start_idx = len(freqs) - 1 - start_idx
            end_idx = len(freqs) - 1 - end_idx
        idx_ranges.append((min(start_idx, end_idx), max(start_idx, end_idx)))

    return lines, idx_ranges


def get_transition(line: Line) -> str:
    transition = line.species_name + " " + f"{line.rest_freq_GHz:.3f}"
    return transition


def make_plot_data(lines: List[Line]) -> Tuple[List[float], List[float], List[float], List[str]]:
    obs_freqs: List[float] = []
    start_freqs: List[float] = []
    end_freqs: List[float] = []
    transitions: List[str] = []
    for line in lines:
        obs_freqs.append(line.observed_freq_GHz)
        start_freqs.append(line.start_freq_GHz)
        end_freqs.append(line.end_freq_GHz)
        transitions.append(get_transition(line))
    return obs_freqs, start_freqs, end_freqs, transitions


def make_plot(
        observation: Observation,
        lsb_rms: float,
        lsb_freqs: np.ndarray,
        lsb_fluxes: np.ndarray,
        lsb_lines: List[Line],
        usb_freqs: np.ndarray,
        usb_lines: List[Line],
        show_only: bool
) -> None:
    lsb_obs_freqs, lsb_start_freqs, lsb_end_freqs, lsb_transitions = \
        make_plot_data(lsb_lines)
    usb_obs_freqs, usb_start_freqs, usb_end_freqs, usb_transitions = \
        make_plot_data(usb_lines)

    fig, ax = plt.subplots()

    # Plot spectrum.
    ax.step(lsb_freqs, lsb_fluxes, lw=0.8, c="black")
    usb_ax = ax.twiny()
    usb_ax.step(usb_freqs, np.ones_like(usb_freqs), c="none")  # Dummy.

    lsb_obs_freq_mark_range = [-7 * lsb_rms, 7 * lsb_rms]
    lsb_start_freq_mark_range = [-8 * lsb_rms, 8 * lsb_rms]
    lsb_end_freq_mark_range = [-8 * lsb_rms, 8 * lsb_rms]
    usb_obs_freq_mark_range = [-7 * lsb_rms, 15 * lsb_rms]
    usb_start_freq_mark_range = [-8 * lsb_rms, 8 * lsb_rms]
    usb_end_freq_mark_range = [-8 * lsb_rms, 8 * lsb_rms]
    if show_only:
        font_size = 8
        y_lim = ()
        lw = 1
    else:
        font_size = 5
        y_max = max(0.3 * np.nanmax(lsb_fluxes), 20 * lsb_rms)
        y_lim = (-10 * lsb_rms, y_max)
        lw = 0.5

    # Plot integration limits.
    for obs_freq, start_freq, end_freq, transition in \
            zip(lsb_obs_freqs, lsb_start_freqs, lsb_end_freqs, lsb_transitions):
        ax.plot([obs_freq, obs_freq], lsb_obs_freq_mark_range, c="blue", lw=lw)
        ax.plot([start_freq, start_freq], lsb_start_freq_mark_range, c="gray", lw=lw)
        ax.plot([end_freq, end_freq], lsb_end_freq_mark_range, c="gray", lw=lw)
        # Note: when "position" is used, the first two coordinates are dummy.
        ax.text(
            0, 0, transition,
            position=(obs_freq, 9 * lsb_rms), ha="center", va="bottom",
            fontsize=font_size, color="blue", rotation=90
        )
    for obs_freq, start_freq, end_freq, transition in \
            zip(usb_obs_freqs, usb_start_freqs, usb_end_freqs, usb_transitions):
        usb_ax.plot([obs_freq, obs_freq], usb_obs_freq_mark_range, c="orange", lw=lw)
        usb_ax.plot([start_freq, start_freq], usb_start_freq_mark_range, c="gray", lw=lw)
        usb_ax.plot([end_freq, end_freq], usb_end_freq_mark_range, c="gray", lw=lw)
        # Note: when "position" is used, the first two coordinates are dummy.
        usb_ax.text(
            0, 0, transition,
            position=(obs_freq, 17 * lsb_rms), ha="center", va="bottom",
            fontsize=font_size, color="orange", rotation=90
        )

    # Axis settings.
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
        f"{observation.object}, "
        f"v_lsr = {observation.vlsr_kms}"
    )

    if show_only:
        plt.show()
    else:
        # todo
        file_path = f"./Figures/CW_Leo/cwleo.{observation.band}.{observation.obs_id}.WBS.sp1.ave.resampled.lines.v{VERSION}.pdf"
        plt.savefig(file_path)
        plt.close()


def get_lines_and_plot_observation(
        observation: Observation,
        use_object_name: bool = True,
        show_only: bool = True
) -> List[Line]:
    base_path = f"Band_{observation.band}/Spectra/{observation.obs_id}"
    lsb_dat_path = f"{base_path}.WBS-LSB.sp1.ave.resampled.dat"
    usb_dat_path = f"{base_path}.WBS-USB.sp1.ave.resampled.dat"

    if not os.path.exists(lsb_dat_path) or not os.path.exists(usb_dat_path):
        return []

    lsb_freqs, lsb_fluxes, lsb_rms = load_spectrum_dat(lsb_dat_path)
    usb_freqs, usb_fluxes, usb_rms = load_spectrum_dat(usb_dat_path)

    object_name = observation.object if use_object_name else None
    lsb_line_ids = find_line_identifications(
        min(lsb_freqs), max(lsb_freqs), object_name
    )
    usb_line_ids = find_line_identifications(
        min(usb_freqs), max(usb_freqs), object_name
    )

    lsb_lines, lsb_idx_ranges = extract_line_data(
        observation, lsb_freqs, lsb_fluxes, lsb_rms, lsb_line_ids, is_lsb=True
    )
    usb_lines, usb_idx_ranges = extract_line_data(
        observation, usb_freqs, usb_fluxes, usb_rms, usb_line_ids, is_lsb=False
    )

    make_plot(
        observation,
        lsb_rms, lsb_freqs, lsb_fluxes, lsb_lines,
        usb_freqs, usb_lines,
        show_only
    )

    lines: List[Line] = []
    lines.extend(lsb_lines)
    lines.extend(usb_lines)

    idx_ranges: List[Tuple[int, int]] = []
    idx_ranges.extend(lsb_idx_ranges)
    idx_ranges.extend(usb_idx_ranges)

    overlapping_dict = find_overlapping_ranges(idx_ranges)
    transitions = [get_transition(line) for line in lines]
    for i, indices in overlapping_dict.items():
        lines[i].blended_transitions = ",".join([transitions[k] for k in indices])

    return lines


def reset_values_for_blended_lines(df: pd.DataFrame) -> pd.DataFrame:
    cond = df["blended_transitions"].str.len() != 0
    for col in ["start_freq_GHz", "end_freq_GHz", "peak_K", "integrated_K_GHz"]:
        assert col in df.columns
        df[col] = np.where(cond, "", df[col])
    return df


def main() -> None:
    all_lines: List[Line] = []
    observations = get_observations()
    for observation in observations:
        if observation.object == "IRC+10216":
            lines = get_lines_and_plot_observation(
                observation, use_object_name=True, show_only=False
            )
            all_lines.extend(lines)

    records = [asdict(line) for line in all_lines]
    df_line = pd.DataFrame.from_records(records)
    df_line = df_line.sort_values(["band", "obs_id", "observed_freq_GHz"])
    df_line = reset_values_for_blended_lines(df_line)
    df_line.to_csv(OUTPUT_TABLE_PATH, sep=";", index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    main()
