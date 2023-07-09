import tempfile
import click
import logging
import os

import atom3.pair as pa
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from project.datasets.analysis.pdb_data import PDBManager
from project.utils.utils import download_pdb_file, gunzip_file


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5']))
def main(output_dir: str, source_type: str):
    logging.info("Analyzing interfaces within each dataset example...")
    with tempfile.TemporaryDirectory() as temp_dir:
        pdb_manager = PDBManager(root_dir=temp_dir)

        if source_type.lower() == "rcsb":
            # Collect (and, if necessary, extract) all training PDB files
            train_pdb_codes = []
            error_pairs = []
            pairs_postprocessed_train_txt = os.path.join(output_dir, 'pairs-postprocessed-train.txt')
            assert os.path.exists(pairs_postprocessed_train_txt), "DB5-Plus train filenames must be curated in advance to partition training and validation filenames."
            with open(pairs_postprocessed_train_txt, "r") as f:
                train_filenames = [line.strip() for line in f.readlines()]
            for train_filename in tqdm(train_filenames):
                try:
                    postprocessed_train_pair: pa.Pair = pd.read_pickle(os.path.join(output_dir, train_filename))
                except Exception as e:
                    logging.error(f"Could not open postprocessed training pair {os.path.join(output_dir, train_filename)} due to: {e}")
                    error_pairs.append(os.path.join(output_dir, train_filename))
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

                l_b_pdb_code = Path(l_b_pdb_filepath).stem + "_" + l_b_df0_chains[0]
                r_b_pdb_code = Path(r_b_pdb_filepath).stem + "_" + r_b_df1_chains[0]
                train_pdb_codes.extend([l_b_pdb_code, r_b_pdb_code])

            # Collect (and, if necessary, extract) all validation PDB files
            val_pdb_codes = []
            pairs_postprocessed_val_txt = os.path.join(output_dir, 'pairs-postprocessed-val.txt')
            assert os.path.exists(pairs_postprocessed_val_txt), "DB5-Plus validation filenames must be curated in advance to partition training and validation filenames."
            with open(pairs_postprocessed_val_txt, "r") as f:
                val_filenames = [line.strip() for line in f.readlines()]
            for val_filename in tqdm(val_filenames):
                try:
                    postprocessed_val_pair: pa.Pair = pd.read_pickle(os.path.join(output_dir, val_filename))
                except Exception as e:
                    logging.error(f"Could not open postprocessed validation pair {os.path.join(output_dir, val_filename)} due to: {e}")
                    error_pairs.append(os.path.join(output_dir, val_filename))
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

                l_b_pdb_code = Path(l_b_pdb_filepath).stem + "_" + l_b_df0_chains[0]
                r_b_pdb_code = Path(r_b_pdb_filepath).stem + "_" + r_b_df1_chains[0]
                val_pdb_codes.extend([l_b_pdb_code, r_b_pdb_code])

            # Record training and validation PDBs as a metadata CSV file
            selected_pdbs_df = pdb_manager.df[pdb_manager.df.id.isin(train_pdb_codes + val_pdb_codes)]
            selected_pdbs_df.to_csv("train_val_pdbs.csv")

            # Plot PDB resolution values versus experiment types
            # Note: the `experiment_type` of `other` indicates an example for which resolution metadata could not successfully be derived
            selected_pdbs_df = selected_pdbs_df[~selected_pdbs_df.experiment_type.isin(["other"])]
            plt.figure(figsize=(10, 6))
            plt.scatter(selected_pdbs_df['resolution'], selected_pdbs_df['experiment_type'])
            plt.xlabel('Resolution')
            plt.ylabel('Experiment Type')
            plt.title('Resolution vs. Experiment Type')
            plt.show()
            plt.savefig("train_val_pdb_resolution_vs_experiment_type.png")

            logging.info("Finished analyzing all training and validation PDB filenames")

        else:
            raise NotImplementedError(f"Source type {source_type} is currently not supported.")


if __name__ == "__main__":
    main()
