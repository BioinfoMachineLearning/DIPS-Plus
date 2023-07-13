import warnings

from pandas.errors import SettingWithCopyWarning

warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=SettingWithCopyWarning)

import click
import graphein
import logging
import loguru
import multiprocessing
import os
import sys

from functools import partial
from pathlib import Path

from project.utils.utils import add_new_feature

GRAPHEIN_FEATURE_NAME_MAPPING = {
    # TODO: Fill out remaining mappings for available Graphein residue-level features
    "expasy_protein_scale": "expasy",
}


@click.command()
@click.argument('raw_data_dir', default='../DIPS/final/raw', type=click.Path(exists=True))
@click.option('--num_cpus', '-c', default=1)
@click.option('--modify_pair_data/--dry_run_only', '-m', default=False)
@click.option('--graphein_feature_to_add', default='expasy_protein_scale', type=str)
def main(raw_data_dir: str, num_cpus: int, modify_pair_data: bool, graphein_feature_to_add: str):
    # Validate requested feature function
    assert (
        hasattr(graphein.protein.features.nodes.amino_acid, graphein_feature_to_add)
    ), f"Graphein must provide the requested node featurization function {graphein_feature_to_add}"

    # Disable DEBUG messages coming from Graphein
    loguru.logger.disable("graphein")
    loguru.logger.remove()
    loguru.logger.add(lambda message: message["level"].name != "DEBUG")

    # Collect paths of files to modify
    raw_data_dir = Path(raw_data_dir)
    raw_data_pickle_filepaths = []
    for root, dirs, files in os.walk(raw_data_dir):
        for dir in dirs:
            for subroot, subdirs, subfiles in os.walk(raw_data_dir / dir):
                for file in subfiles:
                    if file.endswith('.dill'):
                        raw_data_pickle_filepaths.append(raw_data_dir / dir / file)

    # Add to each file the values corresponding to a new feature, using multiprocessing #
    # Define the number of processes to use
    num_processes = min(num_cpus, multiprocessing.cpu_count())
    
    # Split the list of file paths into chunks
    chunk_size = len(raw_data_pickle_filepaths) // num_processes
    file_path_chunks = [
        raw_data_pickle_filepaths[i:i+chunk_size]
        for i in range(0, len(raw_data_pickle_filepaths), chunk_size)
    ]
    assert (
        len(raw_data_pickle_filepaths) == len([fp for chunk in file_path_chunks for fp in chunk])
    ), "Number of input files must match number of files across all file chunks."
    
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Process each chunk of file paths in parallel
    parallel_fn = partial(
        add_new_feature,
        modify_pair_data=modify_pair_data,
        graphein_feature_to_add=graphein_feature_to_add,
        graphein_feature_name_mapping=GRAPHEIN_FEATURE_NAME_MAPPING,
    )
    pool.map(parallel_fn, file_path_chunks)
    
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()


if __name__ == "__main__":
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
