import click
import logging
import os
import warnings

import atom3.pair as pa
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
        train_num_complexes = 0
        train_complex_num_waters = 0
        pairs_postprocessed_train_txt = os.path.join(output_dir, 'pairs-postprocessed-train-before-structure-based-filtering.txt')
        assert os.path.exists(pairs_postprocessed_train_txt), "DIPS-Plus train filenames must be curated in advance."
        with open(pairs_postprocessed_train_txt, "r") as f:
            train_filenames = [line.strip() for line in f.readlines()]
        for train_filename in tqdm(train_filenames):
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

            train_num_complexes += 1

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
                try:
                    target_residue = chain[residue]
                except Exception as e:
                    logging.error(f"Could not locate residue {residue} within chain {chain} for the left-bound training structure {l_b_pdb_filepath} due to: {e}. Skipping...")
                    continue
                target_coords = np.array([atom.get_coord() for atom in target_residue.get_atoms() if atom.get_name() == 'CA']).squeeze()
                interfacing_atoms = l_b_ns.search(target_coords, interfacing_water_distance_cutoff, 'A')
                waters_within_threshold = [atom for atom in interfacing_atoms if atom.get_parent().get_resname() in ['HOH', 'WAT']]
                train_complex_num_waters += len(waters_within_threshold)

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
                try:
                    target_residue = chain[residue]
                except Exception as e:
                    logging.error(f"Could not locate residue {residue} within chain {chain} for the right-bound training structure {r_b_pdb_filepath} due to: {e}. Skipping...")
                    continue
                target_coords = np.array([atom.get_coord() for atom in target_residue.get_atoms() if atom.get_name() == 'CA']).squeeze()
                interfacing_atoms = r_b_ns.search(target_coords, interfacing_water_distance_cutoff, 'A')
                waters_within_threshold = [atom for atom in interfacing_atoms if atom.get_parent().get_resname() in ['HOH', 'WAT']]
                train_complex_num_waters += len(waters_within_threshold)

        # Collect (and, if necessary, extract) all validation PDB files
        val_num_complexes = 0
        val_complex_num_waters = 0
        pairs_postprocessed_val_txt = os.path.join(output_dir, 'pairs-postprocessed-val-before-structure-based-filtering.txt')
        assert os.path.exists(pairs_postprocessed_val_txt), "DIPS-Plus validation filenames must be curated in advance."
        with open(pairs_postprocessed_val_txt, "r") as f:
            val_filenames = [line.strip() for line in f.readlines()]
        for val_filename in tqdm(val_filenames):
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

            val_num_complexes += 1

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
                try:
                    target_residue = chain[residue]
                except Exception as e:
                    logging.error(f"Could not locate residue {residue} within chain {chain} for the left-bound validation structure {l_b_pdb_filepath} due to: {e}. Skipping...")
                    continue
                target_coords = np.array([atom.get_coord() for atom in target_residue.get_atoms() if atom.get_name() == 'CA']).squeeze()
                interfacing_atoms = l_b_ns.search(target_coords, interfacing_water_distance_cutoff, 'A')
                waters_within_threshold = [atom for atom in interfacing_atoms if atom.get_parent().get_resname() in ['HOH', 'WAT']]
                val_complex_num_waters += len(waters_within_threshold)

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
                try:
                    target_residue = chain[residue]
                except Exception as e:
                    logging.error(f"Could not locate residue {residue} within chain {chain} for the right-bound validation structure {r_b_pdb_filepath} due to: {e}. Skipping...")
                    continue
                target_coords = np.array([atom.get_coord() for atom in target_residue.get_atoms() if atom.get_name() == 'CA']).squeeze()
                interfacing_atoms = r_b_ns.search(target_coords, interfacing_water_distance_cutoff, 'A')
                waters_within_threshold = [atom for atom in interfacing_atoms if atom.get_parent().get_resname() in ['HOH', 'WAT']]
                val_complex_num_waters += len(waters_within_threshold)

        # Train complexes
        train_num_waters_per_complex = train_complex_num_waters / train_num_complexes
        logging.info(f"Number of waters, on average, in each training complex: {train_num_waters_per_complex}")

        # Validation complexes
        val_num_waters_per_complex = val_complex_num_waters / val_num_complexes
        logging.info(f"Number of waters, on average, in each validation complex: {val_num_waters_per_complex}")

        # Train + Validation complexes
        train_val_num_waters_per_complex = (train_complex_num_waters + val_complex_num_waters) / (train_num_complexes + val_num_complexes)
        logging.info(f"Number of waters, on average, in each training (or validation) complex: {train_val_num_waters_per_complex}")

        logger.info("Finished analyzing interface waters for all training and validation complexes")

    else:
        raise NotImplementedError(f"Source type {source_type} is currently not supported.")


if __name__ == "__main__":
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
