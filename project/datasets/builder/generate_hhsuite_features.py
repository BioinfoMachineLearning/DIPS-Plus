"""
Source code (MIT-Licensed) inspired by Atom3 (https://github.com/drorlab/atom3 & https://github.com/amorehead/atom3)
"""

import glob
import logging
import os
from os import cpu_count

import click
from atom3.conservation import map_all_profile_hmms
from mpi4py import MPI

from project.utils.utils import get_global_node_rank


@click.command()
@click.argument('pkl_dataset', type=click.Path(exists=True))
@click.argument('pruned_dataset', type=click.Path(exists=True))
@click.argument('hhsuite_db', type=click.Path())
@click.argument('output_dir', type=click.Path())
@click.option('--rank', default=0, type=int)
@click.option('--size', default=1, type=int)
@click.option('--num_cpu_jobs', '-j', default=cpu_count() // 2, type=int)
@click.option('--num_cpus_per_job', '-c', default=2, type=int)
@click.option('--num_iter', '-i', default=2, type=int)
@click.option('--source_type', default='rcsb', type=click.Choice(['rcsb', 'db5', 'evcoupling', 'casp_capri']))
@click.option('--generate_hmm_profile/--generate_msa_only', '-p', default=True)
@click.option('--write_file/--read_file', '-w', default=True)
def main(pkl_dataset: str, pruned_dataset: str, hhsuite_db: str, output_dir: str, rank: int,
         size: int, num_cpu_jobs: int, num_cpus_per_job: int, num_iter: int, source_type: str,
         generate_hmm_profile: bool, write_file: bool):
    """Run external programs for feature generation to turn raw PDB files from (../raw) into sequence or structure-based residue features (saved in ../interim/external_feats by default)."""
    logger = logging.getLogger(__name__)
    logger.info(f'Generating external features from PDB files in {pkl_dataset}')

    # Reestablish global rank
    rank = get_global_node_rank(rank, size)
    logger.info(f"Assigned global rank {rank} of world size {size}")

    # Determine true rank and size for a given node
    bfd_copy_ids = ["_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8",
                    "_9", "_10", "_11", "_12", "_17", "_21", "_25", "_29"]
    bfd_copy_id = bfd_copy_ids[rank]

    # Assemble true ID of the BFD copy to use for generating profile HMMs
    hhsuite_dbs = glob.glob(os.path.join(hhsuite_db + bfd_copy_id, "*bfd*"))
    assert len(hhsuite_dbs) == 1, "Only a single BFD database must be present in the given database directory."
    hhsuite_db = hhsuite_dbs[0]
    logger.info(f'Starting HH-suite for node {rank + 1} out of a global MPI world of size {size},'
                f' with a local MPI world size of {MPI.COMM_WORLD.Get_size()}.'
                f' This node\'s copy of the BFD is {hhsuite_db}')

    # Generate profile HMMs #
    # Run with --write_file=True using one node
    # Then run with --read_file=True using multiple nodes to distribute workload across nodes and their CPU cores
    map_all_profile_hmms(pkl_dataset, pruned_dataset, output_dir, hhsuite_db, num_cpu_jobs,
                         num_cpus_per_job, source_type, num_iter, not generate_hmm_profile, rank, size, write_file)


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
