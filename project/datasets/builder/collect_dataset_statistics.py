import logging
import os

import click

from project.utils.utils import get_global_node_rank, collect_dataset_statistics, DEFAULT_DATASET_STATISTICS


@click.command()
@click.argument('output_dir', default='../DIPS/final/raw', type=click.Path())
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(output_dir: str, rank: int, size: int):
    """Collect all dataset statistics."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info('Aggregating statistics for given dataset')

        # Make sure the output_dir exists
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        # Create dataset statistics CSV if not already existent
        dataset_statistics_csv = os.path.join(output_dir, 'dataset_statistics.csv')
        if not os.path.exists(dataset_statistics_csv):
            # Reset dataset statistics CSV
            with open(dataset_statistics_csv, 'w') as f:
                for key in DEFAULT_DATASET_STATISTICS.keys():
                    f.write("%s, %s\n" % (key, DEFAULT_DATASET_STATISTICS[key]))

        # Aggregate dataset statistics in a readable fashion
        dataset_statistics = collect_dataset_statistics(output_dir)

        # Write out updated dataset statistics
        with open(dataset_statistics_csv, 'w') as f:
            for key in dataset_statistics.keys():
                f.write("%s, %s\n" % (key, dataset_statistics[key]))


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
