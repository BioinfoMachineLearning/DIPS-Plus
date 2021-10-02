import logging
import os
import random
from pathlib import Path

import atom3.pair as pa
import click
import pandas as pd
from atom3 import database as db
from tqdm import tqdm

from project.utils.constants import DB5_TEST_PDB_CODES, ATOM_COUNT_LIMIT
from project.utils.utils import get_global_node_rank


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5', 'evcoupling', 'casp_capri']))
@click.option('--filter_by_atom_count', '-f', default=False)
@click.option('--max_atom_count', '-l', default=ATOM_COUNT_LIMIT)
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(output_dir: str, source_type: str, filter_by_atom_count: bool, max_atom_count: int, rank: int, size: int):
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

        # Separate DB5-test samples from all other DB5 samples
        if source_type.lower() == 'db5':
            # Prepare files
            pairs_postprocessed_test_txt = os.path.join(output_dir, 'pairs-postprocessed-test.txt')
            if not os.path.exists(pairs_postprocessed_test_txt):  # Create test data list if not already existent
                open(pairs_postprocessed_test_txt, 'w').close()
            # Record all DB5-test dataset filenames
            for pair_filename in Path(output_dir).rglob('*.dill'):
                pdb_code = db.get_pdb_code(pair_filename.as_posix())
                for db5_test_pdb_code in DB5_TEST_PDB_CODES:
                    if pdb_code in db5_test_pdb_code:
                        with open(pairs_postprocessed_test_txt, 'a') as f:
                            path, filename = os.path.split(pair_filename.as_posix())
                            filename = os.path.join(pdb_code[1:3], filename)
                            f.write(filename + '\n')  # Pair file was copied


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
