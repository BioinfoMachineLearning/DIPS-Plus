import click
import logging
import os
import warnings

from hickle.lookup import SerializedWarning

warnings.simplefilter("ignore", category=SerializedWarning)

from pathlib import Path
from parallel import submit_jobs

from project.utils.utils import convert_pair_pickle_to_hdf5


@click.command()
@click.argument('raw_data_dir', default='../DIPS/final/raw', type=click.Path(exists=True))
@click.option('--num_cpus', '-c', default=1)
def main(raw_data_dir: str, num_cpus: int):
    raw_data_dir = Path(raw_data_dir)
    raw_data_pickle_filepaths = []
    for root, dirs, files in os.walk(raw_data_dir):
        for dir in dirs:
            for subroot, subdirs, subfiles in os.walk(raw_data_dir / dir):
                for file in subfiles:
                    if file.endswith('.dill'):
                        raw_data_pickle_filepaths.append(raw_data_dir / dir / file)
    inputs = [(pickle_filepath, Path(pickle_filepath).with_suffix(".hdf5")) for pickle_filepath in raw_data_pickle_filepaths]
    submit_jobs(convert_pair_pickle_to_hdf5, inputs, num_cpus)

    # filepath = Path("project/datasets/DIPS/final/raw/0g/10gs.pdb1_0.dill")
    # pickle_example = convert_pair_hdf5_to_pickle(
    #     hdf5_filepath=Path(filepath).with_suffix(".hdf5")
    # )
    # hdf5_file_example = convert_pair_hdf5_to_hdf5_file(
    #     hdf5_filepath=Path(filepath).with_suffix(".hdf5")
    # )
    # print(f"pickle_example: {pickle_example}")
    # print(f"hdf5_file_example: {hdf5_file_example}")


if __name__ == "__main__":
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
