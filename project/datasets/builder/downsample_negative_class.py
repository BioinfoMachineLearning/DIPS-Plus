import logging
import os
from pathlib import Path

import click
from parallel import submit_jobs

from project.utils.utils import get_global_node_rank, downsample_negative_class


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--num_cpus', '-c', default=1)
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5']))
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(output_dir: str, num_cpus: int, source_type: str, rank: int, size: int):
    """Downsample the negative class instances for a given dataset and/or prepackage each complex's labels matrix."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info('Downsampling negative class and/or generating labels for given dataset')

        # Make sure the output_dir exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Collect thread inputs
        inputs = [(pair_filename.as_posix(), pair_filename.as_posix(), source_type)
                  for pair_filename in Path(output_dir).rglob('*.dill')]
        submit_jobs(downsample_negative_class, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
