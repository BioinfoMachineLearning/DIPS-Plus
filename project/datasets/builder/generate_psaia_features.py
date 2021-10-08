"""
Source code (MIT-Licensed) inspired by Atom3 (https://github.com/drorlab/atom3 & https://github.com/amorehead/atom3)
"""

import logging
import os

import click
import atom3.conservation as con

from project.utils.utils import get_global_node_rank


@click.command()
@click.argument('psaia_dir', type=click.Path(exists=True))
@click.argument('psaia_config', type=click.Path(exists=True))
@click.argument('pdb_dataset', type=click.Path(exists=True))
@click.argument('pkl_dataset', type=click.Path(exists=True))
@click.argument('pruned_dataset', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5', 'evcoupling', 'casp_capri']))
@click.option('--rank', '-r', default=0)
@click.option('--size', '-s', default=1)
def main(psaia_dir: str, psaia_config: str, pdb_dataset: str, pkl_dataset: str,
         pruned_dataset: str, output_dir: str, source_type: str, rank: int, size: int):
    """Run external programs for feature generation to turn raw PDB files from (../raw) into sequence or structure-based residue features (saved in ../interim/external_feats by default)."""
    # Reestablish global rank
    rank = get_global_node_rank(rank, size)

    # Ensure that this task only gets run on a single node to prevent race conditions
    if rank == 0:
        logger = logging.getLogger(__name__)
        logger.info(f'Generating PSAIA features from PDB files in {pkl_dataset}')

        # Ensure PSAIA is in PATH
        PSAIA_PATH = psaia_dir
        if PSAIA_PATH not in os.environ["PATH"]:
            logger.info('Adding ' + PSAIA_PATH + ' to system path')
            os.environ["PATH"] += os.path.sep + PSAIA_PATH

        # Generate protrusion indices
        con.map_all_protrusion_indices(psaia_dir, psaia_config, pdb_dataset, pkl_dataset,
                                       pruned_dataset, output_dir, source_type)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
