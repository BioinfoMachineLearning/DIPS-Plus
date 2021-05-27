import logging
import os
from pathlib import Path

import click
from parallel import submit_jobs

from project.utils.utils import get_global_node_rank, add_atoms_to_pairs


@click.command()
@click.argument('pruned_pairs_dir', default='../DIPS/interim/pairs-pruned', type=click.Path(exists=True))
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--num_cpus', '-c', default=1)
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(pruned_pairs_dir: str, output_dir: str, num_cpus: int, rank: int, size: int):
    """Add atoms to complex DataFrames consisting only of residues (i.e. alpha-carbon atoms)."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info('Adding atoms to complex DataFrames')

        # Make sure the output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Collect thread inputs
        inputs = [(pruned_pairs_dir, pair_filename.as_posix(), pair_filename.as_posix())
                  for pair_filename in Path(output_dir).rglob('*.dill')]
        submit_jobs(add_atoms_to_pairs, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
