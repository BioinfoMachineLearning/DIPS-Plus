import glob
import logging
import os

import click
from parallel import submit_jobs

from project.utils.utils import make_dgl_graphs, get_global_node_rank


@click.command()
@click.argument('input_dir', default='../DIPS/final/raw', type=click.Path(exists=True))
@click.argument('output_dir', default='../DIPS/final/processed', type=click.Path())
@click.option('--ext', '-e', default='pt')
@click.option('--num_cpus', '-c', default=1)
@click.option('--edge_dist_cutoff', '-d', default=15.0)
@click.option('--edge_limit', '-l', default=5000)
@click.option('--self_loops', '-s', default=True)
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(input_dir: str, output_dir: str, ext: str, num_cpus: int, edge_dist_cutoff: float,
         edge_limit: int, self_loops: bool, rank: int, size: int):
    """Make DGL graphs out of postprocessed pairs."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info('Creating DGL graphs (final \'processed\' pairs) from postprocessed (final \'raw\') pairs')

        # Make sure the output_dir exists
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        input_files = [file for file in glob.glob(os.path.join(input_dir, '**', '*.dill'), recursive=True)]
        inputs = [(output_dir, input_file, ext, edge_dist_cutoff, edge_limit, self_loops) for input_file in input_files]
        submit_jobs(make_dgl_graphs, inputs, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
