import click
import logging
import multiprocessing
import os

from pathlib import Path

from project.utils.utils import annotate_idr_residues


@click.command()
@click.argument('raw_data_dir', default='../DIPS/final/raw', type=click.Path(exists=True))
@click.option('--num_cpus', '-c', default=1)
def main(raw_data_dir: str, num_cpus: int):
    # Collect paths of files to analyze
    raw_data_dir = Path(raw_data_dir)
    raw_data_pickle_filepaths = []
    for root, dirs, files in os.walk(raw_data_dir):
        for dir in dirs:
            for subroot, subdirs, subfiles in os.walk(raw_data_dir / dir):
                for file in subfiles:
                    if file.endswith('.dill'):
                        raw_data_pickle_filepaths.append(raw_data_dir / dir / file)

    # Annotate whether each residue resides in an IDR, using multiprocessing #
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
    pool.map(annotate_idr_residues, file_path_chunks)
    
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()


if __name__ == "__main__":
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
