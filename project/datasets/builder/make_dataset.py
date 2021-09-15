"""
Source code (MIT-Licensed) originally from DIPS (https://github.com/drorlab/DIPS)
"""

import logging
import os

import atom3.complex as comp
import atom3.neighbors as nb
import atom3.pair as pair
import atom3.parse as pa
import click
from project.utils.utils import get_global_node_rank


@click.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--num_cpus', '-c', default=1)
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
@click.option('--neighbor_def', default='non_heavy_res',
              type=click.Choice(['non_heavy_res', 'non_heavy_atom', 'ca_res', 'ca_atom']))
@click.option('--cutoff', default=6)
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5', 'evcoupling']))
@click.option('--unbound/--bound', default=False)
def main(input_dir: str, output_dir: str, num_cpus: int, rank: int, size: int,
         neighbor_def: str, cutoff: int, source_type: str, unbound: bool):
    """Run data processing scripts to turn raw data from (../raw) into cleaned data ready to be analyzed (saved in ../interim). For reference, pos_idx indicates the IDs of residues in interaction with non-heavy atoms in a cross-protein residue."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info('Making interim data set from raw data')

        parsed_dir = os.path.join(output_dir, 'parsed')
        pa.parse_all(input_dir, parsed_dir, num_cpus)

        complexes_dill = os.path.join(output_dir, 'complexes/complexes.dill')
        comp.complexes(parsed_dir, complexes_dill, source_type)
        pairs_dir = os.path.join(output_dir, 'pairs')
        get_neighbors = nb.build_get_neighbors(neighbor_def, cutoff)
        get_pairs = pair.build_get_pairs(neighbor_def, source_type, unbound, get_neighbors, False)
        complexes = comp.read_complexes(complexes_dill)
        pair.all_complex_to_pairs(complexes, get_pairs, pairs_dir, num_cpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
