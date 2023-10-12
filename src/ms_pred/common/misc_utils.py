""" utils """
import sys
import copy
import logging
from pathlib import Path
import json
from itertools import groupby, islice
from typing import Tuple, List
import pandas as pd
import numpy as np
from tqdm import tqdm

import ms_pred.common.chem_utils as chem_utils

from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.base import rank_zero_experiment


def get_data_dir(dataset_name: str) -> Path:
    return Path("data/spec_datasets") / dataset_name


def setup_logger(save_dir, log_name="output.log", debug=False):
    """Create output directory"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_file = save_dir / log_name

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(level)

    # Define basic logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            file_handler,
        ],
    )

    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))


class ConsoleLogger(LightningLoggerBase):
    """Custom console logger class"""

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self):
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    @rank_zero_experiment
    def version(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        ## No need to log hparams
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):

        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status):
        pass


# Parsing


def parse_spectra(spectra_file: str) -> Tuple[dict, List[Tuple[str, np.ndarray]]]:
    """parse_spectra.

    Parses spectra in the SIRIUS format and returns

    Args:
        spectra_file (str): Name of spectra file to parse
    Return:
        Tuple[dict, List[Tuple[str, np.ndarray]]]: metadata and list of spectra
            tuples containing name and array
    """
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]

    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        # Get spectra
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            # Check if spectra is empty
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                # Add new tuple
                spectras.append((spectra_header, peak_data))
        # Get meta data
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end

            metadata.update(entries)
        group_num += 1

    metadata["_FILE_PATH"] = spectra_file
    metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras


def parse_cfm_out(spectra_file: str, max_merge=False) -> Tuple[dict, pd.DataFrame]:
    """parse_cfm_out.

    Args:
        out_file (str): out_file
        max_merge (bool): If true, merge across energies

    Returns:
        dict, pd.DataFrame:
    """
    lines = [i.strip() for i in open(spectra_file, "r").readlines()]
    specs, keys = "\n".join(lines).split("\n\n")

    # Step 1: Process keys
    key_dict = {}
    for row in keys.split("\n"):
        row = row.strip()
        num, mass, smi = row.split()
        key_dict[num] = dict(mass=mass, smi=smi)

    lines = specs.split("\n")
    my_iterator = groupby(lines, lambda line: line.startswith("#"))
    meta_groups, spec_groups = [(i[0], list(i[1])) for i in my_iterator]

    # Step 2: extract meta
    meta_data = {}
    for i in meta_groups[1]:
        if "=" not in i:
            continue
        key, val = i.split("=", 1)
        meta_data[key[1:]] = val

    # Step 3: extract specs
    new_iter = groupby(spec_groups[1], lambda x: x.startswith("energy"))
    all_specs = {list(i)[0]: list(next(new_iter)[1]) for _, i in new_iter}

    # Combine all
    full_spec = []
    for spec_key, spec_vals in all_specs.items():
        key_to_amt = {}
        for spec_val in spec_vals:
            spec_val = spec_val.replace(")", "")
            mass, inten, rest = spec_val.split(" ", 2)
            num, amts = rest.split(" (")
            num_to_amt = dict(zip(num.split(), map(float, amts.split())))
            key_to_amt.update(num_to_amt)

        # Spec format is going to be
        for k, amt in key_to_amt.items():
            k_info = key_dict.get(k)
            mass = k_info["mass"]
            smi = k_info["smi"]
            chem_form = chem_utils.form_from_smi(smi)
            new_info = dict(
                smi=smi, mass=mass, form=chem_form, inten=amt, energy=spec_key
            )
            full_spec.append(new_info)
    full_spec = pd.DataFrame(full_spec)
    if max_merge:
        full_spec = full_spec.groupby("smi").max().reset_index()

    # Safe sub h
    sub_h = lambda x: chem_utils.formula_difference(x, "H") if "H" in x else x
    less_h = [sub_h(i) for i in full_spec["form"].values]
    full_spec["form_no_h"] = less_h
    full_spec["formula_mass_no_adduct"] = [chem_utils.formula_mass(i) for i in less_h]
    full_spec["ionization"] = "[M+H]+"
    return meta_data, full_spec


def process_spec_file(meta, tuples, precision=4):
    """process_spec_file."""

    parent_mass = meta.get("parentmass", None)
    if parent_mass is None:
        print(f"missing parentmass for spec")
        parent_mass = 1000000

    parent_mass = float(parent_mass)

    # First norm spectra
    fused_tuples = [x for _, x in tuples if x.size > 0]

    if len(fused_tuples) == 0:
        return

    mz_to_inten_pair = {}
    new_tuples = []
    for i in fused_tuples:
        for tup in i:
            mz, inten = tup
            mz_ind = np.round(mz, precision)
            cur_pair = mz_to_inten_pair.get(mz_ind)
            if cur_pair is None:
                mz_to_inten_pair[mz_ind] = tup
                new_tuples.append(tup)
            elif inten > cur_pair[1]:
                cur_pair[1] = inten
            else:
                pass

    merged_spec = np.vstack(new_tuples)
    merged_spec = merged_spec[merged_spec[:, 0] <= (parent_mass + 1)]
    merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()

    # Sqrt intensities here
    merged_spec[:, 1] = np.sqrt(merged_spec[:, 1])
    return merged_spec


def bin_form_file(spec_file, num_bins, upper_limit) -> Tuple[dict, np.ndarray]:
    """bin_form_file.

    Args:
        spec_file:
        num_bins:
        upper_limit:

    Returns:
        Tuple[dict, np.ndarray]:
    """

    loaded_json = json.load(open(spec_file, "r"))
    if loaded_json["output_tbl"] is None:
        return {}, None

    # Load without adduct involved
    mz = loaded_json["output_tbl"]["formula_mass_no_adduct"]
    inten = loaded_json["output_tbl"]["ms2_inten"]

    # Don't renorm; already procesed prior!
    spec_ar = np.vstack([mz, inten]).transpose(1, 0)
    binned = bin_spectra([spec_ar], num_bins, upper_limit)
    # normed = common.norm_spectrum(binned)
    avged = binned.mean(0)
    return {}, avged


def max_inten_spec(spec, max_num_inten: int = 60, inten_thresh: float = 0):
    """max_inten_spec.

    Args:
        spec: 2D spectra array
        max_num_inten: Max number of peaks
        inten_thresh: Min intensity to alloow in returned peak

    Return:
        Spec filtered down


    """
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]

    # Make sure to only take max of each formula
    # Sort by intensity and select top subpeaks
    new_sort_order = np.argsort(spec_intens)[::-1]
    if max_num_inten is not None:
        new_sort_order = new_sort_order[:max_num_inten]

    spec_masses = spec_masses[new_sort_order]
    spec_intens = spec_intens[new_sort_order]

    spec_mask = spec_intens > inten_thresh
    spec_masses = spec_masses[spec_mask]
    spec_intens = spec_intens[spec_mask]
    spec = np.vstack([spec_masses, spec_intens]).transpose(1, 0)
    return spec


def norm_spectrum(binned_spec: np.ndarray) -> np.ndarray:
    """norm_spectrum.

    Normalizes each spectral channel to have norm 1
    This change is made in place

    Args:
        binned_spec (np.ndarray) : Vector of spectras

    Return:
        np.ndarray where each channel has max(1)
    """

    spec_maxes = binned_spec.max(1)

    non_zero_max = spec_maxes > 0

    spec_maxes = spec_maxes[non_zero_max]
    binned_spec[non_zero_max] = binned_spec[non_zero_max] / spec_maxes.reshape(-1, 1)

    # Add in sqrt
    binned_spec = np.sqrt(binned_spec)

    return binned_spec


def bin_spectra(
    spectras: List[np.ndarray],
    num_bins: int = 15000,
    upper_limit: int = 1500,
    pool_fn: str = "max",
) -> np.ndarray:
    """bin_spectra.

    Args:
        spectras (List[np.ndarray]): Input list of spectra tuples
            [(header, spec array)]
        num_bins (int): Number of discrete bins from [0, upper_limit)
        upper_limit (int): Max m/z to consider featurizing
        pool_fn (str): Pooling function to use for binning (max or add)

    Return:
        np.ndarray of shape [channels, num_bins]
    """
    if pool_fn == "add":
        pool = lambda x, y: x + y
    elif pool_fn == "max":
        pool = lambda x, y: max(x, y)
    else:
        raise NotImplementedError()

    bins = np.linspace(0, upper_limit, num=num_bins)
    binned_spec = np.zeros((len(spectras), len(bins)))
    for spec_index, spec in enumerate(spectras):

        # Convert to digitized spectra
        digitized_mz = np.digitize(spec[:, 0], bins=bins)

        # Remove all spectral peaks out of range
        in_range = digitized_mz < len(bins)
        digitized_mz, spec = digitized_mz[in_range], spec[in_range, :]

        # Add the current peaks to the spectra
        # Use a loop rather than vectorize because certain bins have conflicts
        # based upon resolution
        for bin_index, spec_val in zip(digitized_mz, spec[:, 1]):
            cur_val = binned_spec[spec_index, bin_index]
            binned_spec[spec_index, bin_index] = pool(spec_val, cur_val)

    return binned_spec


def digitize_ar(
    ar: np.ndarray, num_bins: int = 15000, upper_limit: int = 1500
) -> np.ndarray:
    """digitize_ar.

    Args:
        ar (np.ndarray): ar
        num_bins (int): Num bins
        upper_limit (int): upper lim

    Return:
        np ndarray containing indices
    """
    bins = np.linspace(0, upper_limit, num=num_bins)
    return np.digitize(ar, bins=bins)


def bin_mass_results(
    mass,
    mass_bins=[
        (0, 200),
        (200, 300),
        (300, 400),
        (400, 500),
        (500, 600),
        (600, 700),
        (700, 2000),
    ],
):
    """bin_mass_results.

    Use to stratify results

    Args:
        mass:
        mass_bins:
    """
    for i, j in mass_bins:
        m_str = f"{i} - {j}"
        if mass <= j and mass > i:
            return m_str


def batches(it, chunk_size: int):
    """Consume an iterable in batches of size chunk_size""" ""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])


def batches_num_chunks(it, num_chunks: int):
    """Consume an iterable in batches of size chunk_size""" ""
    chunk_size = len(it) // num_chunks
    return batches(it, chunk_size)


def build_mgf_str(
    meta_spec_list: List[Tuple[dict, List[Tuple[str, np.ndarray]]]],
    merge_charges=True,
    parent_mass_keys=["PEPMASS", "parentmass", "PRECURSOR_MZ"],
) -> str:
    """build_mgf_str.

    Args:
        meta_spec_list (List[Tuple[dict, List[Tuple[str, np.ndarray]]]]): meta_spec_list

    Returns:
        str:
    """
    entries = []
    for meta, spec in tqdm(meta_spec_list):
        str_rows = ["BEGIN IONS"]

        # Try to add precusor mass
        for i in parent_mass_keys:
            if i in meta:
                pep_mass = float(meta.get(i, -100))
                str_rows.append(f"PEPMASS={pep_mass}")
                break

        for k, v in meta.items():
            str_rows.append(f"{k.upper().replace(' ', '_')}={v}")

        if merge_charges:
            spec_ar = np.vstack([i[1] for i in spec])
            spec_ar = np.vstack([i for i in sorted(spec_ar, key=lambda x: x[0])])
        else:
            raise NotImplementedError()
        str_rows.extend([f"{i} {j}" for i, j in spec_ar])
        str_rows.append("END IONS")

        str_out = "\n".join(str_rows)
        entries.append(str_out)

    full_out = "\n\n".join(entries)
    return full_out
