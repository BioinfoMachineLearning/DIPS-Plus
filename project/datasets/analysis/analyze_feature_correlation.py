import click
import logging
import os

import atom3.pair as pa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm

from project.utils.utils import download_pdb_file, gunzip_file


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5']))
@click.option('--feature_types_to_correlate', default='rcsb', type=click.Choice(['rsa_value-rd_value', 'rsa_value-cn_value', 'rd_value-cn_value']))
def main(output_dir: str, source_type: str, feature_types_to_correlate: str):
    logger = logging.getLogger(__name__)
    logger.info("Analyzing feature correlation for each dataset example...")

    features_to_correlate = feature_types_to_correlate.split("-")
    assert len(features_to_correlate) == 2, "Exactly two features may be currently compared for correlation measures."

    if source_type.lower() == "rcsb":
        # Collect (and, if necessary, extract) all training PDB files
        train_feature_values = []
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

            l_b_df0_feature_values = postprocessed_train_pair.df0[features_to_correlate].applymap(lambda x: np.nan if x == 'NA' else x).dropna().apply(pd.to_numeric)
            r_b_df1_feature_values = postprocessed_train_pair.df1[features_to_correlate].applymap(lambda x: np.nan if x == 'NA' else x).dropna().apply(pd.to_numeric)
            train_feature_values.append(pd.concat([l_b_df0_feature_values, r_b_df1_feature_values]))

        # Collect (and, if necessary, extract) all validation PDB files
        val_feature_values = []
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

            l_b_df0_feature_values = postprocessed_val_pair.df0[features_to_correlate].applymap(lambda x: np.nan if x == 'NA' else x).dropna().apply(pd.to_numeric)
            r_b_df1_feature_values = postprocessed_val_pair.df1[features_to_correlate].applymap(lambda x: np.nan if x == 'NA' else x).dropna().apply(pd.to_numeric)
            val_feature_values.append(pd.concat([l_b_df0_feature_values, r_b_df1_feature_values]))

        # Train PDBs
        train_feature_values_df = pd.concat(train_feature_values)
        train_feature_values_correlation, train_feature_values_p_value = pearsonr(train_feature_values_df[features_to_correlate[0]], train_feature_values_df[features_to_correlate[1]])
        logger.info(f"With a p-value of {train_feature_values_p_value}, the Pearson's correlation of feature values `{feature_types_to_correlate}` within the training dataset is: {train_feature_values_correlation}")
        train_joint_plot = sns.jointplot(x=features_to_correlate[0], y=features_to_correlate[1], data=train_feature_values_df, kind='hex')
        # Add correlation value to the jointplot
        train_joint_plot.ax_joint.annotate(f"Training correlation: {train_feature_values_correlation:.2f}", xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=12)
        # Save the jointplot
        train_joint_plot.savefig(f"train_{feature_types_to_correlate}_correlation.png")
        plt.close()

        # Validation PDBs
        val_feature_values_df = pd.concat(val_feature_values)
        val_feature_values_correlation, val_feature_values_p_value = pearsonr(val_feature_values_df[features_to_correlate[0]], val_feature_values_df[features_to_correlate[1]])
        logger.info(f"With a p-value of {val_feature_values_p_value}, the Pearson's correlation of feature values `{feature_types_to_correlate}` within the validation dataset is: {val_feature_values_correlation}")
        val_joint_plot = sns.jointplot(x=features_to_correlate[0], y=features_to_correlate[1], data=val_feature_values_df, kind='hex')
        # Add correlation value to the jointplot
        val_joint_plot.ax_joint.annotate(f"Validation correlation: {val_feature_values_correlation:.2f}", xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=12)
        # Save the jointplot
        val_joint_plot.savefig(f"val_{feature_types_to_correlate}_correlation.png")
        plt.close()

        # Train + Validation PDBs
        train_val_feature_values_df = pd.concat([train_feature_values_df, val_feature_values_df])
        train_val_feature_values_correlation, train_val_feature_values_p_value = pearsonr(train_val_feature_values_df[features_to_correlate[0]], train_val_feature_values_df[features_to_correlate[1]])
        logger.info(f"With a p-value of {train_val_feature_values_p_value}, the Pearson's correlation of feature values `{feature_types_to_correlate}` within the training and validation dataset is: {train_val_feature_values_correlation}")
        train_val_joint_plot = sns.jointplot(x=features_to_correlate[0], y=features_to_correlate[1], data=train_val_feature_values_df, kind='hex')
        # Add correlation value to the jointplot
        train_val_joint_plot.ax_joint.annotate(f"Training + Validation Correlation: {train_val_feature_values_correlation:.2f}", xy=(0.5, 0.95), xycoords='axes fraction', ha='center', fontsize=12)
        # Save the jointplot
        train_val_joint_plot.savefig(f"train_val_{feature_types_to_correlate}_correlation.png")
        plt.close()

        logger.info("Finished analyzing feature correlation for all training and validation PDBs")

    else:
        raise NotImplementedError(f"Source type {source_type} is currently not supported.")


if __name__ == "__main__":
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
