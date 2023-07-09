import tempfile
import click
import logging
import os

import atom3.pair as pa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from project.datasets.analysis.pdb_data import PDBManager
from project.utils.utils import download_pdb_file, gunzip_file


def plot_mean_std_freq(means, stds, freqs, title):
    plt.figure(figsize=(10, 6))
    num_experiments = len(means)
    bar_width = 0.4
    index = np.arange(num_experiments)
    
    # Plot bars for mean values with customized colors
    plt.barh(index, means, height=bar_width, color='salmon', label='Mean')
    
    # Plot error bars representing standard deviation with customized colors
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.errorbar(mean, i, xerr=std, color='red')
        
        # Plot vertical lines for the first and second standard deviation ranges with customized colors
        plt.vlines([mean - std, mean + std], i - bar_width/2, i + bar_width/2, color='royalblue', linestyles='dashed', label='1st Std Dev')
        plt.vlines([mean - 2*std, mean + 2*std], i - bar_width/2, i + bar_width/2, color='limegreen', linestyles='dotted', label='2nd Std Dev')
    
    plt.yticks(index, means.index)
    plt.xlabel('Resolution')
    plt.ylabel('Experiment Type')
    plt.title(title)
    plt.legend(loc="upper left")

    # Sort means, stds, and freqs based on the experiment types
    means = means[means.index.sort_values()]
    stds = stds[stds.index.sort_values()]
    freqs = freqs[freqs.index.sort_values()]
    freqs = (freqs / freqs.sum()) * 100
    
    # Calculate middle points of each bar
    middles = means

    # Calculate the visual center of each bar
    visual_center = middles / 2
    
    # Add frequency (count) of each experiment type at the middle point of each bar
    for i, freq in enumerate(freqs):
        plt.text(visual_center[i], i, f"{freq:.4f}%", va='center', ha='center', color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(title.lower().replace(' ', '_') + ".png")


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5']))
def main(output_dir: str, source_type: str):
    logger = logging.getLogger(__name__)
    logger.info("Analyzing experiment types and resolution for each dataset example...")

    if source_type.lower() == "rcsb":
        train_metadata_csv_filepath = os.path.join(output_dir, "train_pdb_metadata.csv")
        val_metadata_csv_filepath = os.path.join(output_dir, "val_pdb_metadata.csv")
        train_val_metadata_csv_filepath = os.path.join(output_dir, "train_val_pdb_metadata.csv")
        metadata_csv_filepaths = [train_metadata_csv_filepath, val_metadata_csv_filepath, train_val_metadata_csv_filepath]

        if any(not os.path.exists(fp) for fp in metadata_csv_filepaths):
            with tempfile.TemporaryDirectory() as temp_dir:
                pdb_manager = PDBManager(root_dir=temp_dir)

                # Collect (and, if necessary, extract) all training PDB files
                train_pdb_codes = []
                pairs_postprocessed_train_txt = os.path.join(output_dir, 'pairs-postprocessed-train-before-structure-based-filtering.txt')
                assert os.path.exists(pairs_postprocessed_train_txt), "DB5-Plus train filenames must be curated in advance to partition training and validation filenames."
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

                    l_b_pdb_code = Path(l_b_pdb_filepath).stem + "_" + l_b_df0_chains[0]
                    r_b_pdb_code = Path(r_b_pdb_filepath).stem + "_" + r_b_df1_chains[0]
                    train_pdb_codes.extend([l_b_pdb_code, r_b_pdb_code])

                # Collect (and, if necessary, extract) all validation PDB files
                val_pdb_codes = []
                pairs_postprocessed_val_txt = os.path.join(output_dir, 'pairs-postprocessed-val-before-structure-based-filtering.txt')
                assert os.path.exists(pairs_postprocessed_val_txt), "DB5-Plus validation filenames must be curated in advance to partition training and validation filenames."
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

                    l_b_pdb_code = Path(l_b_pdb_filepath).stem + "_" + l_b_df0_chains[0]
                    r_b_pdb_code = Path(r_b_pdb_filepath).stem + "_" + r_b_df1_chains[0]
                    val_pdb_codes.extend([l_b_pdb_code, r_b_pdb_code])

                # Record training and validation PDBs as a metadata CSV file
                train_pdbs_df = pdb_manager.df[pdb_manager.df.id.isin(train_pdb_codes)]
                train_pdbs_df.to_csv(train_metadata_csv_filepath)

                val_pdbs_df = pdb_manager.df[pdb_manager.df.id.isin(val_pdb_codes)]
                val_pdbs_df.to_csv(val_metadata_csv_filepath)

                train_val_pdbs_df = pdb_manager.df[pdb_manager.df.id.isin(train_pdb_codes + val_pdb_codes)]
                train_val_pdbs_df.to_csv(train_val_metadata_csv_filepath)

        assert all(os.path.exists(fp) for fp in metadata_csv_filepaths), "To analyze RCSB complexes, the corresponding metadata must previously have been collected."
        train_pdbs_df = pd.read_csv(train_metadata_csv_filepath, index_col=0)
        val_pdbs_df = pd.read_csv(val_metadata_csv_filepath, index_col=0)
        train_val_pdbs_df = pd.read_csv(train_val_metadata_csv_filepath, index_col=0)

        # Train PDBs
        train_pdbs_df = train_pdbs_df[~train_pdbs_df.experiment_type.isin(["other"])]
        means_train = train_pdbs_df.groupby('experiment_type')['resolution'].mean()
        stds_train = train_pdbs_df.groupby('experiment_type')['resolution'].std()
        freqs_train = train_pdbs_df['experiment_type'].value_counts()
        plot_mean_std_freq(means_train, stds_train, freqs_train, 'Resolution vs. Experiment Type (Train)')

        # Validation PDBs
        val_pdbs_df = val_pdbs_df[~val_pdbs_df.experiment_type.isin(["other"])]
        means_val = val_pdbs_df.groupby('experiment_type')['resolution'].mean()
        stds_val = val_pdbs_df.groupby('experiment_type')['resolution'].std()
        freqs_val = val_pdbs_df['experiment_type'].value_counts()
        plot_mean_std_freq(means_val, stds_val, freqs_val, 'Resolution vs. Experiment Type (Validation)')

        # Train + Validation PDBs
        train_val_pdbs_df = train_val_pdbs_df[~train_val_pdbs_df.experiment_type.isin(["other"])]
        means_train_val = train_val_pdbs_df.groupby('experiment_type')['resolution'].mean()
        stds_train_val = train_val_pdbs_df.groupby('experiment_type')['resolution'].std()
        freqs_train_val = train_val_pdbs_df['experiment_type'].value_counts()
        plot_mean_std_freq(means_train_val, stds_train_val, freqs_train_val, 'Resolution vs. Experiment Type (Train + Validation)')

        logger.info("Finished analyzing experiment types and resolution for all training and validation PDBs")

    else:
        raise NotImplementedError(f"Source type {source_type} is currently not supported.")


if __name__ == "__main__":
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
