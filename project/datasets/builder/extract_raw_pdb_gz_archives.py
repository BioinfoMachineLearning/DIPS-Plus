import gzip
import logging
import os
import shutil

import click
from project.utils.utils import get_global_node_rank


@click.command()
@click.argument('gz_data_dir', type=click.Path(exists=True))
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(gz_data_dir: str, rank: int, size: int):
    """ Run GZ extraction logic to turn raw data from (raw/) into extracted data ready to be analyzed by DSSP."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info('Extracting raw GZ archives')

        # Iterate directory structure, extracting all GZ archives along the way
        data_dir = os.path.abspath(gz_data_dir) + '/'
        raw_pdb_list = os.listdir(data_dir)
        for pdb_dir in raw_pdb_list:
            for pdb_gz in os.listdir(data_dir + pdb_dir):
                if pdb_gz.endswith('.gz'):
                    _, ext = os.path.splitext(pdb_gz)
                    gzip_dir = data_dir + pdb_dir + '/' + pdb_gz
                    extract_dir = data_dir + pdb_dir + '/' + _
                    if not os.path.exists(extract_dir):
                        with gzip.open(gzip_dir, 'rb') as f_in:
                            with open(extract_dir, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
