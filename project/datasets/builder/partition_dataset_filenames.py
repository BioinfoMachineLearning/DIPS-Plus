import logging
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Union

import atom3.pair as pa
import click
import pandas as pd
from atom3 import database as db
from tqdm import tqdm

from project.utils.constants import DB5_TEST_PDB_CODES, ATOM_COUNT_LIMIT
from project.utils.utils import download_pdb_file, get_global_node_rank


FOLDSEEK_DEFAULT_TAB_COLUMNS = [
    "query",
    "target",
    "fident",
    "alnlen",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "evalue",
    "bits",
    "prob",
    "lddt",
    "alntmscore",
]


def gunzip_file(file_path):
    try:
        subprocess.run(['gunzip', file_path], check=True)
    except subprocess.CalledProcessError as e:
        print('gunzip error:', e)


def partition_filenames_by_sequence(output_dir: str, source_type: str, filter_by_atom_count: bool, max_atom_count: int):
    pairs_postprocessed_txt = os.path.join(output_dir, 'pairs-postprocessed.txt')
    open(pairs_postprocessed_txt, 'w').close()  # Create pairs-postprocessed.txt from scratch each run

    # Record dataset filenames conditionally by sequence length (if requested - otherwise, record all)
    pair_filenames = [pair_filename for pair_filename in Path(output_dir).rglob('*.dill')]
    for pair_filename in tqdm(pair_filenames):
        struct_id = pair_filename.as_posix().split(os.sep)[-2]
        if filter_by_atom_count and source_type.lower() == 'rcsb':
            postprocessed_pair: pa.Pair = pd.read_pickle(pair_filename)
            if len(postprocessed_pair.df0) < max_atom_count and len(postprocessed_pair.df1) < max_atom_count:
                with open(pairs_postprocessed_txt, 'a') as f:
                    path, filename = os.path.split(pair_filename.as_posix())
                    filename = os.path.join(struct_id, filename)
                    f.write(filename + '\n')  # Pair file was copied
        else:
            with open(pairs_postprocessed_txt, 'a') as f:
                path, filename = os.path.split(pair_filename.as_posix())
                filename = os.path.join(struct_id, filename)
                f.write(filename + '\n')  # Pair file was copied

    # Separate training samples from validation samples
    if source_type.lower() == 'rcsb':
        # Prepare files
        pairs_postprocessed_train_txt = os.path.join(output_dir, 'pairs-postprocessed-train.txt')
        if not os.path.exists(pairs_postprocessed_train_txt):  # Create train data list if not already existent
            open(pairs_postprocessed_train_txt, 'w').close()
        pairs_postprocessed_val_txt = os.path.join(output_dir, 'pairs-postprocessed-val.txt')
        if not os.path.exists(pairs_postprocessed_val_txt):  # Create val data list if not already existent
            open(pairs_postprocessed_val_txt, 'w').close()
        # Write out training-validation partitions for DIPS
        output_dirs = [filename
                        for filename in os.listdir(output_dir)
                        if os.path.isdir(os.path.join(output_dir, filename))]
        # Get training and validation directories separately
        num_train_dirs = int(0.8 * len(output_dirs))
        train_dirs = random.sample(output_dirs, num_train_dirs)
        val_dirs = list(set(output_dirs) - set(train_dirs))
        # Ascertain training and validation filename separately
        filenames_frame = pd.read_csv(pairs_postprocessed_txt, header=None)
        train_filenames = [os.path.join(train_dir, filename)
                            for train_dir in train_dirs
                            for filename in os.listdir(os.path.join(output_dir, train_dir))
                            if os.path.join(train_dir, filename) in filenames_frame.values]
        val_filenames = [os.path.join(val_dir, filename)
                            for val_dir in val_dirs
                            for filename in os.listdir(os.path.join(output_dir, val_dir))
                            if os.path.join(val_dir, filename) in filenames_frame.values]
        # Create separate .txt files to describe the training list and validation list, respectively
        train_filenames_frame, val_filenames_frame = pd.DataFrame(train_filenames), pd.DataFrame(val_filenames)
        train_filenames_frame.to_csv(pairs_postprocessed_train_txt, header=None, index=None, sep=' ', mode='a')
        val_filenames_frame.to_csv(pairs_postprocessed_val_txt, header=None, index=None, sep=' ', mode='a')

    # Separate DB5-Plus-test samples from all other DB5-Plus samples
    if source_type.lower() == 'db5':
        # Prepare files
        pairs_postprocessed_test_txt = os.path.join(output_dir, 'pairs-postprocessed-test.txt')
        if not os.path.exists(pairs_postprocessed_test_txt):  # Create test data list if not already existent
            open(pairs_postprocessed_test_txt, 'w').close()
        # Record all DB5-Plus-test dataset filenames
        for pair_filename in Path(output_dir).rglob('*.dill'):
            pdb_code = db.get_pdb_code(pair_filename.as_posix())
            for db5_test_pdb_code in DB5_TEST_PDB_CODES:
                if pdb_code in db5_test_pdb_code:
                    with open(pairs_postprocessed_test_txt, 'a') as f:
                        path, filename = os.path.split(pair_filename.as_posix())
                        filename = os.path.join(pdb_code[1:3], filename)
                        f.write(filename + '\n')  # Pair file was copied


def copy_pdb_chain_dicts_to_temp_directory(pdb_chain_dicts: List[Dict[str, Any]], temp_dir_name: str):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    # Create the destination directory structure in the temp directory
    destination_dir = os.path.join(temp_dir, temp_dir_name)
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate over the list of file dictionaries
    for pdb_chain_dict in pdb_chain_dicts:
        l_pdb_filepath = pdb_chain_dict["l_pdb_filepath"]
        r_pdb_filepath = pdb_chain_dict["r_pdb_filepath"]
        l_chain = pdb_chain_dict["l_chain"]
        r_chain = pdb_chain_dict["r_chain"]
        temp_l_pdb_filepath = os.path.join(destination_dir, os.path.basename(l_pdb_filepath))
        temp_r_pdb_filepath = os.path.join(destination_dir, os.path.basename(r_pdb_filepath))

        # Extract files using dict #
        # Copy the left PDB file to the temp directory
        shutil.copy2(l_pdb_filepath, destination_dir)
        if l_chain is not None:  # Note: if `lchain` is `None`, we write out all of the original left chains
            command = f"pdb_selchain -{l_chain} {temp_l_pdb_filepath} > {temp_l_pdb_filepath}.tmp"
            subprocess.run(command, shell=True)
            shutil.move(f"{temp_l_pdb_filepath}.tmp", temp_l_pdb_filepath)
        # Copy the right PDB file to the temp directory
        shutil.copy2(r_pdb_filepath, destination_dir)
        if r_chain is not None:  # Note: if `rchain` is `None`, we write out all of the original right chains
            command = f"pdb_selchain -{r_chain} {temp_r_pdb_filepath} > {temp_r_pdb_filepath}.tmp"
            subprocess.run(command, shell=True)
            shutil.move(f"{temp_r_pdb_filepath}.tmp", temp_r_pdb_filepath)

    return temp_dir


def run_foldseek_easy_search(input_directory, reference_directory, alignment_filepath, temp_directory, sensitivity_threshold: float = 1e-3):
    # Run FoldSeek's easy-search CLI module
    command = [
        'foldseek',
        'easy-search',
        input_directory,
        reference_directory,
        alignment_filepath,
        temp_directory,
        "--exhaustive-search",
        "--format-output",
        ",".join(FOLDSEEK_DEFAULT_TAB_COLUMNS),
        "-e",
        str(sensitivity_threshold)
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command executed successfully
    if result.returncode != 0:
        print(result.stderr)
        raise Exception('Error occurred while running FoldSeek\'s easy-search.')

    # Create a DataFrame from the alignment data
    df = pd.read_csv(alignment_filepath, sep="\t", names=FOLDSEEK_DEFAULT_TAB_COLUMNS)
    return df


def partition_filenames_by_structure(
    output_dir: str,
    source_type: str,
    filter_by_atom_count: bool,
    max_atom_count: int,
    partition_criterion: str,
    partition_criterion_threshold: Union[int, float],
    partition_criterion_search_sensitivity: float,
    overwrite_existing_pairs_postprocessed: bool = False,
):
    pairs_postprocessed_txt = os.path.join(output_dir, 'pairs-postprocessed.txt')
    if not os.path.exists(pairs_postprocessed_txt) or (overwrite_existing_pairs_postprocessed and os.path.exists(pairs_postprocessed_txt)):
        open(pairs_postprocessed_txt, 'w').close()  # Create pairs-postprocessed.txt from scratch each run

        # Record dataset filenames conditionally by sequence length (if requested - otherwise, record all)
        pair_filenames = [pair_filename for pair_filename in Path(output_dir).rglob('*.dill')]
        for pair_filename in tqdm(pair_filenames):
            struct_id = pair_filename.as_posix().split(os.sep)[-2]
            if filter_by_atom_count and source_type.lower() == 'rcsb':
                postprocessed_pair: pa.Pair = pd.read_pickle(pair_filename)
                if len(postprocessed_pair.df0) < max_atom_count and len(postprocessed_pair.df1) < max_atom_count:
                    with open(pairs_postprocessed_txt, 'a') as f:
                        path, filename = os.path.split(pair_filename.as_posix())
                        filename = os.path.join(struct_id, filename)
                        f.write(filename + '\n')  # Pair file was copied
            else:
                with open(pairs_postprocessed_txt, 'a') as f:
                    path, filename = os.path.split(pair_filename.as_posix())
                    filename = os.path.join(struct_id, filename)
                    f.write(filename + '\n')  # Pair file was copied

    # Separate DB5-Plus-test samples from all other DB5-Plus samples
    pairs_postprocessed_test_txt = os.path.join(output_dir, 'pairs-postprocessed-test.txt')
    if source_type.lower() == "db5":
        if not os.path.exists(pairs_postprocessed_test_txt) or (overwrite_existing_pairs_postprocessed and os.path.exists(pairs_postprocessed_test_txt)):
            # e.g., Create test data list if not already existent
            open(pairs_postprocessed_test_txt, 'w').close()
        # Record all DB5-Plus-test dataset filenames
        for pair_filename in Path(output_dir).rglob('*.dill'):
            pdb_code = db.get_pdb_code(pair_filename.as_posix())
            for db5_test_pdb_code in DB5_TEST_PDB_CODES:
                if pdb_code in db5_test_pdb_code:
                    with open(pairs_postprocessed_test_txt, 'a') as f:
                        path, filename = os.path.split(pair_filename.as_posix())
                        filename = os.path.join(pdb_code[1:3], filename)
                        f.write(filename + '\n')  # Pair file was copied

    # Separate DIPS-Plus training samples from validation samples
    elif source_type.lower() == "rcsb":
        # Prepare DIPS-Plus files
        creating_data_list = False
        pairs_postprocessed_train_txt = os.path.join(output_dir, 'pairs-postprocessed-train.txt')
        if not os.path.exists(pairs_postprocessed_train_txt) or (overwrite_existing_pairs_postprocessed and os.path.exists(pairs_postprocessed_train_txt)):
            # e.g., Create train data list if not already existent
            creating_data_list = True
            open(pairs_postprocessed_train_txt, 'w').close()
        pairs_postprocessed_val_txt = os.path.join(output_dir, 'pairs-postprocessed-val.txt')
        if not os.path.exists(pairs_postprocessed_val_txt) or (overwrite_existing_pairs_postprocessed and os.path.exists(pairs_postprocessed_val_txt)):
            # e.g., Create val data list if not already existent
            creating_data_list = True
            open(pairs_postprocessed_val_txt, 'w').close()

        if creating_data_list:
            # Write out initial training-validation partitions for DIPS-Plus
            output_dirs = [filename
                            for filename in os.listdir(output_dir)
                            if os.path.isdir(os.path.join(output_dir, filename))]
            # Get training and validation directories separately
            num_train_dirs = int(0.8 * len(output_dirs))
            train_dirs = random.sample(output_dirs, num_train_dirs)
            val_dirs = list(set(output_dirs) - set(train_dirs))
            # Ascertain training and validation filename separately
            filenames_frame = pd.read_csv(pairs_postprocessed_txt, header=None)
            train_filenames = [os.path.join(train_dir, filename)
                                for train_dir in train_dirs
                                for filename in os.listdir(os.path.join(output_dir, train_dir))
                                if os.path.join(train_dir, filename) in filenames_frame.values]
            val_filenames = [os.path.join(val_dir, filename)
                                for val_dir in val_dirs
                                for filename in os.listdir(os.path.join(output_dir, val_dir))
                                if os.path.join(val_dir, filename) in filenames_frame.values]
            # Create separate .txt files to describe the training list and validation list, respectively
            train_filenames_frame, val_filenames_frame = pd.DataFrame(train_filenames), pd.DataFrame(val_filenames)
            train_filenames_frame.to_csv(pairs_postprocessed_train_txt, header=None, index=None, sep=' ', mode='a')
            val_filenames_frame.to_csv(pairs_postprocessed_val_txt, header=None, index=None, sep=' ', mode='a')

        # Collect (and, if necessary, extract) all training PDB files
        train_entries = []
        assert os.path.exists(pairs_postprocessed_train_txt), "DB5-Plus train filenames must be curated in advance to partition training and validation filenames."
        with open(pairs_postprocessed_train_txt, "r") as f:
            train_filenames = [line.strip() for line in f.readlines()]
        for train_filename in train_filenames:
            postprocessed_train_pair: pa.Pair = pd.read_pickle(os.path.join(output_dir, train_filename))
            pdb_code = postprocessed_train_pair.df0.pdb_name[0].split("_")[0][1:3]
            pdb_dir = os.path.join(Path(output_dir).parent.parent, "raw", "pdb", pdb_code)
            l_b_pdb_filepath = os.path.join(pdb_dir, postprocessed_train_pair.df0.pdb_name[0])
            r_b_pdb_filepath = os.path.join(pdb_dir, postprocessed_train_pair.df1.pdb_name[0])
            assert (
                len(postprocessed_train_pair.df0.pdb_name.unique()) == len(postprocessed_train_pair.df0.chain.unique()) == 1
            ), "Only a single PDB filename and chain identifier can be associated with a single training example."
            assert (
                len(postprocessed_train_pair.df1.pdb_name.unique()) == len(postprocessed_train_pair.df1.chain.unique()) == 1
            ), "Only a single PDB filename and chain identifier can be associated with a single training example."
            train_entry = {
                "l_pdb_filepath": l_b_pdb_filepath,
                "r_pdb_filepath": r_b_pdb_filepath,
                "l_chain": postprocessed_train_pair.df0.chain[0],
                "r_chain": postprocessed_train_pair.df1.chain[0],
            }
            train_entries.append(train_entry)
            if not os.path.exists(l_b_pdb_filepath) and os.path.exists(l_b_pdb_filepath + ".gz"):
                gunzip_file(l_b_pdb_filepath)
            if not os.path.exists(r_b_pdb_filepath) and os.path.exists(r_b_pdb_filepath + ".gz"):
                gunzip_file(r_b_pdb_filepath)
            if not os.path.exists(l_b_pdb_filepath):
                download_pdb_file(os.path.basename(l_b_pdb_filepath), l_b_pdb_filepath)
            if not os.path.exists(r_b_pdb_filepath):
                download_pdb_file(os.path.basename(r_b_pdb_filepath), r_b_pdb_filepath)
            assert os.path.exists(l_b_pdb_filepath) and os.path.exists(r_b_pdb_filepath), "Both left and right-bound PDB files collected must exist."

        # Collect (and, if necessary, extract) all validation PDB files
        val_entries = []
        assert os.path.exists(pairs_postprocessed_val_txt), "DB5-Plus validation filenames must be curated in advance to partition training and validation filenames."
        with open(pairs_postprocessed_val_txt, "r") as f:
            val_filenames = [line.strip() for line in f.readlines()]
        for val_filename in val_filenames[:100]:
            postprocessed_val_pair: pa.Pair = pd.read_pickle(os.path.join(output_dir, val_filename))
            pdb_code = postprocessed_val_pair.df0.pdb_name[0].split("_")[0][1:3]
            pdb_dir = os.path.join(Path(output_dir).parent.parent, "raw", "pdb", pdb_code)
            l_b_pdb_filepath = os.path.join(pdb_dir, postprocessed_val_pair.df0.pdb_name[0])
            r_b_pdb_filepath = os.path.join(pdb_dir, postprocessed_val_pair.df1.pdb_name[0])
            assert (
                len(postprocessed_val_pair.df0.pdb_name.unique()) == len(postprocessed_val_pair.df0.chain.unique()) == 1
            ), "Only a single PDB filename and chain identifier can be associated with a single validation example."
            assert (
                len(postprocessed_val_pair.df1.pdb_name.unique()) == len(postprocessed_val_pair.df1.chain.unique()) == 1
            ), "Only a single PDB filename and chain identifier can be associated with a single validation example."
            val_entry = {
                "l_pdb_filepath": l_b_pdb_filepath,
                "r_pdb_filepath": r_b_pdb_filepath,
                "l_chain": postprocessed_val_pair.df0.chain[0],
                "r_chain": postprocessed_val_pair.df1.chain[0],
            }
            val_entries.append(val_entry)
            if not os.path.exists(l_b_pdb_filepath) and os.path.exists(l_b_pdb_filepath + ".gz"):
                gunzip_file(l_b_pdb_filepath)
            if not os.path.exists(r_b_pdb_filepath) and os.path.exists(r_b_pdb_filepath + ".gz"):
                gunzip_file(r_b_pdb_filepath)
            if not os.path.exists(l_b_pdb_filepath):
                download_pdb_file(os.path.basename(l_b_pdb_filepath), l_b_pdb_filepath)
            if not os.path.exists(r_b_pdb_filepath):
                download_pdb_file(os.path.basename(r_b_pdb_filepath), r_b_pdb_filepath)
            assert os.path.exists(l_b_pdb_filepath) and os.path.exists(r_b_pdb_filepath), "Both left and right-bound PDB files collected must exist."

        # Collect all test PDB filepaths
        pairs_postprocessed_test_txt = pairs_postprocessed_test_txt.replace("DIPS", "DB5")
        assert os.path.exists(pairs_postprocessed_test_txt), "DB5-Plus test filenames must be organized in advance to create training and validation filenames."
        with open(pairs_postprocessed_test_txt, "r") as f:
            test_filenames = [line.strip() for line in f.readlines()]
        test_entries = []
        for test_filename in test_filenames:
            postprocessed_test_pair: pa.Pair = pd.read_pickle(os.path.join(output_dir.replace("DIPS", "DB5"), test_filename))
            pdb_code = postprocessed_test_pair.df0.pdb_name[0].split("_")[0]
            pdb_dir = os.path.join(Path(output_dir).parent.as_posix().replace("DIPS", "DB5").replace("final", "raw"), pdb_code)
            l_u_pdb_filepath = os.path.join(pdb_dir, postprocessed_test_pair.df0.pdb_name[0])
            r_u_pdb_filepath = os.path.join(pdb_dir, postprocessed_test_pair.df1.pdb_name[0])
            assert (
                len(postprocessed_test_pair.df0.pdb_name.unique()) == 1
            ), "Only a single PDB filename can be associated with a single test example."
            assert (
                len(postprocessed_test_pair.df1.pdb_name.unique()) == 1
            ), "Only a single PDB filename can be associated with a single test example."
            test_entry = {
                "l_pdb_filepath": l_u_pdb_filepath,
                "r_pdb_filepath": r_u_pdb_filepath,
                "l_chain": None,
                "r_chain": None,
            }
            test_entries.append(test_entry)
            assert os.path.exists(l_u_pdb_filepath) and os.path.exists(r_u_pdb_filepath), "Both left and right-unbound PDB files collected must exist."

        # Organize all training, validation, and test PDB files in a shared temporary directory
        train_temp_dir = copy_pdb_chain_dicts_to_temp_directory(train_entries, temp_dir_name="train")
        val_temp_dir = copy_pdb_chain_dicts_to_temp_directory(val_entries, temp_dir_name="val")
        test_temp_dir = copy_pdb_chain_dicts_to_temp_directory(test_entries, temp_dir_name="test")
        
        # Run FoldSeek on each collection of PDB files
        train_alignment_df = run_foldseek_easy_search(
            input_directory=os.path.join(train_temp_dir, "train"),
            reference_directory=os.path.join(test_temp_dir, "test"),
            alignment_filepath=os.path.join(train_temp_dir, "aln"),
            temp_directory=os.path.join(train_temp_dir, "tmp_artifacts"),
            sensitivity_threshold=partition_criterion_search_sensitivity,
        )
        val_alignment_df = run_foldseek_easy_search(
            input_directory=os.path.join(val_temp_dir, "val"),
            reference_directory=os.path.join(test_temp_dir, "test"),
            alignment_filepath=os.path.join(val_temp_dir, "aln"),
            temp_directory=os.path.join(val_temp_dir, "tmp_artifacts"),
            sensitivity_threshold=partition_criterion_search_sensitivity,
        )
        test_alignment_df = run_foldseek_easy_search(
            input_directory=os.path.join(test_temp_dir, "test"),
            reference_directory=os.path.join(test_temp_dir, "test"),
            alignment_filepath=os.path.join(test_temp_dir, "aln"),
            temp_directory=os.path.join(test_temp_dir, "tmp_artifacts"),
            sensitivity_threshold=partition_criterion_search_sensitivity,
        )

        # Filter each collection of filepaths based on the results from FoldSeek,
        # driven by the user's choice of partitioning criterion and search sensitivity
        train_violations_df = (
            train_alignment_df[train_alignment_df[partition_criterion] >= partition_criterion_threshold]
        )
        val_violations_df = (
            val_alignment_df[val_alignment_df[partition_criterion] >= partition_criterion_threshold]
        )
        test_violations_df = test_alignment_df[
            (test_alignment_df[partition_criterion] >= partition_criterion_threshold) & \
            (test_alignment_df["query"] != test_alignment_df["target"])
        ]

        # Note: For multi-model PDB files, we need to `split` off the model indices within their filenames and simply remove all of such models
        train_filenames_to_remove = train_violations_df["query"].apply(lambda x: x.strip().split("_")[0]).unique().tolist()
        val_filenames_to_remove = val_violations_df["query"].apply(lambda x: x.strip().split("_")[0]).unique().tolist()
        test_filenames_to_remove = test_violations_df["query"].apply(lambda x: x.strip().split("_")[0]).unique().tolist()

        train_filenames_to_remove = [os.path.join(filename[1:3], filename) for filename in train_filenames_to_remove]
        val_filenames_to_remove = [os.path.join(filename[1:3], filename) for filename in val_filenames_to_remove]
        test_filenames_to_remove = [os.path.join(filename[1:3], filename) for filename in test_filenames_to_remove]

        # Load the training, validation, and test filepaths
        train_filenames_frame = pd.read_csv(pairs_postprocessed_train_txt, header=None, index_col=None, sep=' ')
        val_filenames_frame = pd.read_csv(pairs_postprocessed_val_txt, header=None, index_col=None, sep=' ')
        test_filenames_frame = pd.read_csv(pairs_postprocessed_test_txt, header=None, index_col=None, sep=' ')

        # Identify, using substring matching, which rows in the original filenames lists are in violation of structure-based filters
        train_filenames_to_keep_frame = train_filenames_frame[~train_filenames_frame[0].str.contains("|".join(train_filenames_to_remove))]
        val_filenames_to_keep_frame = val_filenames_frame[~val_filenames_frame[0].str.contains("|".join(val_filenames_to_remove))]
        test_filenames_to_keep_frame = test_filenames_frame[~test_filenames_frame[0].str.contains("|".join(test_filenames_to_remove))]

        # Record paths to structurally-filtered files
        train_filenames_to_keep_frame.to_csv(pairs_postprocessed_train_txt, header=None, index=None, sep=' ', mode='w')
        val_filenames_to_keep_frame.to_csv(pairs_postprocessed_val_txt, header=None, index=None, sep=' ', mode='w')
        # test_filenames_to_keep_frame.to_csv(pairs_postprocessed_test_txt, header=None, index=None, sep=' ', mode='w')
        
    else:
        raise NotImplementedError(f"Source type {source_type} is currently not supported.")


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5', 'evcoupling', 'casp_capri']))
@click.option('--filter_by_atom_count/--bypass_atom_count_filtering', '-f', default=False)
@click.option('--max_atom_count', '-l', default=ATOM_COUNT_LIMIT)
@click.option('--partition_modality', '-p', default="structure")
@click.option('--partition_criterion', '-p', default="prob")
@click.option('--partition_criterion_threshold', '-t', default=0.5)
@click.option('--partition_criterion_search_sensitivity', '-e', default=0.1)
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(
    output_dir: str,
    source_type: str,
    filter_by_atom_count: bool,
    max_atom_count: int,
    partition_modality: str,
    partition_criterion: str,
    partition_criterion_threshold: float,
    partition_criterion_search_sensitivity: float,
    rank: int,
    size: int,
):
    """Partition dataset filenames."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info(f'Writing filename DataFrames to their respective text files')

        # Make sure the output_dir exists
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if partition_modality == "sequence":
            partition_filenames_by_sequence(output_dir, source_type, filter_by_atom_count, max_atom_count)
        elif partition_modality == "structure":
            assert partition_criterion in FOLDSEEK_DEFAULT_TAB_COLUMNS, "The selected structure-based partitioning criterion must be included in FoldSeek's default output fields."
            partition_filenames_by_structure(output_dir, source_type, filter_by_atom_count, max_atom_count, partition_criterion, partition_criterion_threshold, partition_criterion_search_sensitivity)
        else:
            raise NotImplementedError(f"Partition modality {partition_modality} is currently not supported.")


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
