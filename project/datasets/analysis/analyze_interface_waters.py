import click
import logging
import os
import warnings

import atom3.pair as pa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Bio import BiopythonWarning
from Bio.PDB import NeighborSearch
from Bio.PDB import PDBParser
from pathlib import Path
from tqdm import tqdm

from project.utils.utils import download_pdb_file, gunzip_file


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5']))
@click.option('--interfacing_water_distance_cutoff', default=10.0, type=float)
def main(output_dir: str, source_type: str, interfacing_water_distance_cutoff: float):
    logger = logging.getLogger(__name__)
    logger.info("Analyzing interface waters within each dataset example...")

    if source_type.lower() == "rcsb":
        parser = PDBParser()

        # Filter and suppress BioPython warnings
        warnings.filterwarnings("ignore", category=BiopythonWarning)

        # Collect (and, if necessary, extract) all training PDB files
        train_complex_num_waters = []
        pairs_postprocessed_train_txt = os.path.join(output_dir, 'pairs-postprocessed-train-before-structure-based-filtering.txt')
        assert os.path.exists(pairs_postprocessed_train_txt), "DIPS-Plus train filenames must be curated in advance."
        with open(pairs_postprocessed_train_txt, "r") as f:
            train_filenames = [line.strip() for line in f.readlines()]
        for train_filename in tqdm(train_filenames):
            complex_num_waters = 0
            try:
                postprocessed_train_pair: pa.Pair = pd.read_pickle(os.path.join(output_dir, train_filename))
            except Exception as e:
                logging.error(f"Could not open postprocessed training pair {os.path.join(output_dir, train_filename)} due to: {e}")
                continue
            pdb_code = postprocessed_train_pair.df0.pdb_name[0].split("_")[0][1:3]
            pdb_dir = os.path.join(Path(output_dir).parent.parent, "raw", "pdb", pdb_code)
            l_b_pdb_filepath = os.path.join(pdb_dir, postprocessed_train_pair.df0.pdb_name[0])
            r_b_pdb_filepath = os.path.join(pdb_dir, postprocessed_train_pair.df1.pdb_name[0])
            l_b_df0_chains = postprocessed_train_pair.df0.chain.unique()
            r_b_df1_chains = postprocessed_train_pair.df1.chain.unique()
            assert (
                len(postprocessed_train_pair.df0.pdb_name.unique()) == len(l_b_df0_chains) == 1
            ), "Only a single PDB filename and chain identifier can be associated with a single training example."
            assert (
                len(postprocessed_train_pair.df1.pdb_name.unique()) == len(r_b_df1_chains) == 1
            ), "Only a single PDB filename and chain identifier can be associated with a single training example."
            if not os.path.exists(l_b_pdb_filepath) and os.path.exists(l_b_pdb_filepath + ".gz"):
                gunzip_file(l_b_pdb_filepath)
            if not os.path.exists(r_b_pdb_filepath) and os.path.exists(r_b_pdb_filepath + ".gz"):
                gunzip_file(r_b_pdb_filepath)
            if not os.path.exists(l_b_pdb_filepath):
                download_pdb_file(os.path.basename(l_b_pdb_filepath), l_b_pdb_filepath)
            if not os.path.exists(r_b_pdb_filepath):
                download_pdb_file(os.path.basename(r_b_pdb_filepath), r_b_pdb_filepath)
            assert os.path.exists(l_b_pdb_filepath) and os.path.exists(r_b_pdb_filepath), "Both left and right-bound PDB files collected must exist."

            l_b_structure = parser.get_structure('protein', l_b_pdb_filepath)
            r_b_structure = parser.get_structure('protein', r_b_pdb_filepath)

            l_b_interface_residues = postprocessed_train_pair.df0[postprocessed_train_pair.df0.index.isin(postprocessed_train_pair.pos_idx[:, 0])]
            r_b_interface_residues = postprocessed_train_pair.df1[postprocessed_train_pair.df1.index.isin(postprocessed_train_pair.pos_idx[:, 1])]

            try:
                l_b_ns = NeighborSearch(list(l_b_structure.get_atoms()))
                for index, row in l_b_interface_residues.iterrows():
                    chain_id = row['chain']
                    residue = row['residue'].strip()
                    model = l_b_structure[0]
                    chain = model[chain_id]
                    if residue.lstrip("-").isdigit():
                        residue = int(residue)
                    else:
                        residue_index, residue_icode = residue[:-1], residue[-1:]
                        if residue_icode.strip() == "":
                            residue = int(residue)
                        else:
                            residue = (" ", int(residue_index), residue_icode)
                    target_residue = chain[residue]
                    target_coords = np.array([atom.get_coord() for atom in target_residue.get_atoms() if atom.get_name() == 'CA']).squeeze()
                    interfacing_atoms = l_b_ns.search(target_coords, interfacing_water_distance_cutoff, 'A')
                    waters_within_threshold = [atom for atom in interfacing_atoms if atom.get_parent().get_resname() in ['HOH', 'WAT']]
                    complex_num_waters += len(waters_within_threshold)
            except Exception as e:
                logging.error(f"Unable to locate interface waters for left-bound training structure {l_b_pdb_filepath} due to: {e}. Skipping...")
                continue

            try:
                r_b_ns = NeighborSearch(list(r_b_structure.get_atoms()))
                for index, row in r_b_interface_residues.iterrows():
                    chain_id = row['chain']
                    residue = row['residue'].strip()
                    model = r_b_structure[0]
                    chain = model[chain_id]
                    if residue.lstrip("-").isdigit():
                        residue = int(residue)
                    else:
                        residue_index, residue_icode = residue[:-1], residue[-1:]
                        residue = (" ", int(residue_index), residue_icode)
                    target_residue = chain[residue]
                    target_coords = np.array([atom.get_coord() for atom in target_residue.get_atoms() if atom.get_name() == 'CA']).squeeze()
                    interfacing_atoms = r_b_ns.search(target_coords, interfacing_water_distance_cutoff, 'A')
                    waters_within_threshold = [atom for atom in interfacing_atoms if atom.get_parent().get_resname() in ['HOH', 'WAT']]
                    complex_num_waters += len(waters_within_threshold)
            except Exception as e:
                logging.error(f"Unable to locate interface waters for right-bound training structure {r_b_pdb_filepath} due to: {e}. Skipping...")
                continue

            train_complex_num_waters.append(complex_num_waters)

        # Collect (and, if necessary, extract) all validation PDB files
        val_complex_num_waters = []
        pairs_postprocessed_val_txt = os.path.join(output_dir, 'pairs-postprocessed-val-before-structure-based-filtering.txt')
        assert os.path.exists(pairs_postprocessed_val_txt), "DIPS-Plus validation filenames must be curated in advance."
        with open(pairs_postprocessed_val_txt, "r") as f:
            val_filenames = [line.strip() for line in f.readlines()]
        for val_filename in tqdm(val_filenames):
            complex_num_waters = 0
            try:
                postprocessed_val_pair: pa.Pair = pd.read_pickle(os.path.join(output_dir, val_filename))
            except Exception as e:
                logging.error(f"Could not open postprocessed validation pair {os.path.join(output_dir, val_filename)} due to: {e}")
                continue
            pdb_code = postprocessed_val_pair.df0.pdb_name[0].split("_")[0][1:3]
            pdb_dir = os.path.join(Path(output_dir).parent.parent, "raw", "pdb", pdb_code)
            l_b_pdb_filepath = os.path.join(pdb_dir, postprocessed_val_pair.df0.pdb_name[0])
            r_b_pdb_filepath = os.path.join(pdb_dir, postprocessed_val_pair.df1.pdb_name[0])
            l_b_df0_chains = postprocessed_val_pair.df0.chain.unique()
            r_b_df1_chains = postprocessed_val_pair.df1.chain.unique()
            assert (
                len(postprocessed_val_pair.df0.pdb_name.unique()) == len(l_b_df0_chains) == 1
            ), "Only a single PDB filename and chain identifier can be associated with a single validation example."
            assert (
                len(postprocessed_val_pair.df1.pdb_name.unique()) == len(r_b_df1_chains) == 1
            ), "Only a single PDB filename and chain identifier can be associated with a single validation example."
            if not os.path.exists(l_b_pdb_filepath) and os.path.exists(l_b_pdb_filepath + ".gz"):
                gunzip_file(l_b_pdb_filepath)
            if not os.path.exists(r_b_pdb_filepath) and os.path.exists(r_b_pdb_filepath + ".gz"):
                gunzip_file(r_b_pdb_filepath)
            if not os.path.exists(l_b_pdb_filepath):
                download_pdb_file(os.path.basename(l_b_pdb_filepath), l_b_pdb_filepath)
            if not os.path.exists(r_b_pdb_filepath):
                download_pdb_file(os.path.basename(r_b_pdb_filepath), r_b_pdb_filepath)
            assert os.path.exists(l_b_pdb_filepath) and os.path.exists(r_b_pdb_filepath), "Both left and right-bound PDB files collected must exist."

            l_b_structure = parser.get_structure('protein', l_b_pdb_filepath)
            r_b_structure = parser.get_structure('protein', r_b_pdb_filepath)

            l_b_interface_residues = postprocessed_val_pair.df0[postprocessed_val_pair.df0.index.isin(postprocessed_val_pair.pos_idx[:, 0])]
            r_b_interface_residues = postprocessed_val_pair.df1[postprocessed_val_pair.df1.index.isin(postprocessed_val_pair.pos_idx[:, 1])]

            try:
                l_b_ns = NeighborSearch(list(l_b_structure.get_atoms()))
                for index, row in l_b_interface_residues.iterrows():
                    chain_id = row['chain']
                    residue = row['residue'].strip()
                    model = l_b_structure[0]
                    chain = model[chain_id]
                    if residue.lstrip("-").isdigit():
                        residue = int(residue)
                    else:
                        residue_index, residue_icode = residue[:-1], residue[-1:]
                        residue = (" ", int(residue_index), residue_icode)
                    target_residue = chain[residue]
                    target_coords = np.array([atom.get_coord() for atom in target_residue.get_atoms() if atom.get_name() == 'CA']).squeeze()
                    interfacing_atoms = l_b_ns.search(target_coords, interfacing_water_distance_cutoff, 'A')
                    waters_within_threshold = [atom for atom in interfacing_atoms if atom.get_parent().get_resname() in ['HOH', 'WAT']]
                    complex_num_waters += len(waters_within_threshold)
            except Exception as e:
                logging.error(f"Unable to locate interface waters for left-bound validation structure {l_b_pdb_filepath} due to: {e}. Skipping...")
                continue

            try:
                r_b_ns = NeighborSearch(list(r_b_structure.get_atoms()))
                for index, row in r_b_interface_residues.iterrows():
                    chain_id = row['chain']
                    residue = row['residue'].strip()
                    model = r_b_structure[0]
                    chain = model[chain_id]
                    if residue.lstrip("-").isdigit():
                        residue = int(residue)
                    else:
                        residue_index, residue_icode = residue[:-1], residue[-1:]
                        residue = (" ", int(residue_index), residue_icode)
                    target_residue = chain[residue]
                    target_coords = np.array([atom.get_coord() for atom in target_residue.get_atoms() if atom.get_name() == 'CA']).squeeze()
                    interfacing_atoms = r_b_ns.search(target_coords, interfacing_water_distance_cutoff, 'A')
                    waters_within_threshold = [atom for atom in interfacing_atoms if atom.get_parent().get_resname() in ['HOH', 'WAT']]
                    complex_num_waters += len(waters_within_threshold)
            except Exception as e:
                logging.error(f"Unable to locate interface waters for right-bound validation structure {r_b_pdb_filepath} due to: {e}. Skipping...")
                continue

            val_complex_num_waters.append(complex_num_waters)

        train_val_complex_num_waters = train_complex_num_waters + val_complex_num_waters

        # Calculate mean values
        training_mean = np.mean(train_complex_num_waters)
        validation_mean = np.mean(val_complex_num_waters)
        training_validation_mean = np.mean(train_val_complex_num_waters)

        # Plotting the distributions
        plt.figure(figsize=(10, 6))  # Set the size of the figure

        # Training data distribution
        plt.subplot(131)  # 1 row, 3 columns, plot 1 (leftmost)
        plt.hist(train_complex_num_waters, bins=10, color='royalblue')
        plt.axvline(training_mean, color='limegreen', linestyle='dashed', linewidth=2)
        plt.text(training_mean + 0.1, plt.ylim()[1] * 0.9, f'   Mean: {training_mean:.2f}', color='limegreen')
        plt.title('Train Interface Waters')
        plt.xlabel('Count')
        plt.ylabel('Frequency')

        # Validation data distribution
        plt.subplot(132)  # 1 row, 3 columns, plot 2 (middle)
        plt.hist(val_complex_num_waters, bins=10, color='royalblue')
        plt.axvline(validation_mean, color='limegreen', linestyle='dashed', linewidth=2)
        plt.text(validation_mean + 0.1, plt.ylim()[1] * 0.9, f'   Mean: {validation_mean:.2f}', color='limegreen')
        plt.title('Validation Interface Waters')
        plt.xlabel('Count')
        plt.ylabel('Frequency')

        # Combined data distribution
        plt.subplot(133)  # 1 row, 3 columns, plot 3 (rightmost)
        plt.hist(train_val_complex_num_waters, bins=10, color='royalblue')
        plt.axvline(training_validation_mean, color='limegreen', linestyle='dashed', linewidth=2)
        plt.text(training_validation_mean + 0.1, plt.ylim()[1] * 0.9, f'   Mean: {training_validation_mean:.2f}', color='limegreen')
        plt.title('Train+Validation Interface Waters')
        plt.xlabel('Count')
        plt.ylabel('Frequency')

        plt.tight_layout()  # Adjust the spacing between subplots
        plt.show()  # Display the plots
        plt.savefig("train_val_interface_waters_analysis.png")

        logger.info("Finished analyzing interface waters for all training and validation complexes")

    else:
        raise NotImplementedError(f"Source type {source_type} is currently not supported.")


if __name__ == "__main__":
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
