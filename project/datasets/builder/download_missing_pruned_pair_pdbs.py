import logging
import os

import click
from atom3.database import get_structures_filenames, get_pdb_name, get_pdb_code
from atom3.utils import slice_list
from mpi4py import MPI
from parallel import submit_jobs

from project.utils.utils import get_global_node_rank, download_missing_pruned_pair_pdbs


@click.command()
@click.argument('output_dir', default='../DIPS/raw/pdb', type=click.Path(exists=True))
@click.argument('pruned_pairs_dir', default='../DIPS/interim/pairs-pruned', type=click.Path(exists=True))
@click.option('--num_cpus', '-c', default=1)
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(output_dir: str, pruned_pairs_dir: str, num_cpus: int, rank: int, size: int):
    """Download missing pruned pair PDB files."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)
    logger = logging.getLogger(__name__)
    logger.info(f'Beginning missing PDB downloads for node {rank + 1} out of a global MPI world of size {size},'
                f' with a local MPI world size of {MPI.COMM_WORLD.Get_size()}')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Get work filenames
    logger.info(f'Looking for all pairs in {pruned_pairs_dir}')
    requested_filenames = get_structures_filenames(pruned_pairs_dir, extension='.dill')
    requested_filenames = [filename for filename in requested_filenames]
    requested_keys = [get_pdb_name(x) for x in requested_filenames]
    work_keys = [key for key in requested_keys]
    work_filenames = [os.path.join(pruned_pairs_dir, get_pdb_code(work_key)[1:3], work_key + '.dill')
                      for work_key in work_keys]
    logger.info(f'Found {len(work_keys)} work pair(s) in {pruned_pairs_dir}')

    # Reserve an equally-sized portion of the full work load for a given rank in the MPI world
    work_filenames = list(set(work_filenames))  # Remove any duplicate filenames
    work_filename_rank_batches = slice_list(work_filenames, size)
    work_filenames = work_filename_rank_batches[rank]

    # Collect thread inputs
    inputs = [(logger, output_dir, work_filename) for work_filename in work_filenames]
    submit_jobs(download_missing_pruned_pair_pdbs, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
