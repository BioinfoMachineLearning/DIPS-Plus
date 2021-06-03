import logging
import os

import click
from atom3.database import get_structures_filenames, get_pdb_name, get_pdb_code
from atom3.utils import slice_list
from mpi4py import MPI
from parallel import submit_jobs

from project.utils.utils import get_global_node_rank
from project.utils.utils import postprocess_pruned_pairs


@click.command()
@click.argument('raw_pdb_dir', default='../DIPS/raw/pdb', type=click.Path(exists=True))
@click.argument('pruned_pairs_dir', default='../DIPS/interim/pairs-pruned', type=click.Path(exists=True))
@click.argument('external_feats_dir', default='../DIPS/interim/external_feats', type=click.Path(exists=True))
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--num_cpus', '-c', default=1)
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5']))
def main(raw_pdb_dir: str, pruned_pairs_dir: str, external_feats_dir: str, output_dir: str,
         num_cpus: int, rank: int, size: int, source_type: str):
    """Run postprocess_pruned_pairs on all provided complexes."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)
    logger = logging.getLogger(__name__)
    logger.info(f'Starting postprocessing for node {rank + 1} out of a global MPI world of size {size},'
                f' with a local MPI world size of {MPI.COMM_WORLD.Get_size()}')

    # Make sure the output_dir exists
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get work filenames
    logger.info(f'Looking for all pairs in {pruned_pairs_dir}')
    requested_filenames = get_structures_filenames(pruned_pairs_dir, extension='.dill')
    requested_filenames = [filename for filename in requested_filenames]
    requested_keys = [get_pdb_name(x) for x in requested_filenames]
    produced_filenames = get_structures_filenames(output_dir, extension='.dill')
    produced_keys = [get_pdb_name(x) for x in produced_filenames]
    work_keys = [key for key in requested_keys if key not in produced_keys]
    rscb_pruned_pair_ext = '.dill' if source_type.lower() == 'rcsb' else ''
    work_filenames = [os.path.join(pruned_pairs_dir, get_pdb_code(work_key)[1:3], work_key + rscb_pruned_pair_ext)
                      for work_key in work_keys]
    logger.info(f'Found {len(work_keys)} work pair(s) in {pruned_pairs_dir}')

    # Reserve an equally-sized portion of the full work load for a given rank in the MPI world
    work_filenames = list(set(work_filenames))  # Remove any duplicate filenames
    work_filename_rank_batches = slice_list(work_filenames, size)
    work_filenames = work_filename_rank_batches[rank]

    # Get filenames in which our threads will store output
    output_filenames = []
    for pdb_filename in work_filenames:
        sub_dir = output_dir + '/' + get_pdb_code(pdb_filename)[1:3]
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        new_output_filename = sub_dir + '/' + get_pdb_name(pdb_filename) + ".dill" if \
            source_type == 'rcsb' else \
            sub_dir + '/' + get_pdb_name(pdb_filename)
        output_filenames.append(new_output_filename)

    # Collect thread inputs
    inputs = [(raw_pdb_dir, external_feats_dir, i, o, source_type)
              for i, o in zip(work_filenames, output_filenames)]
    submit_jobs(postprocess_pruned_pairs, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
