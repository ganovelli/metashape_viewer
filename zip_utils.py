import zipfile
import tempfile
import os
from typing import Iterable, Tuple
import shutil

def rmdir_if_exists(path: str):
    """Remove directory if it exists."""
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        
def extract_paths_to_tempdir(
    zip_path: str,
    internal_paths: Iterable[str],
) -> Tuple[str, dict]:
    """
    Extract selected files from a ZIP into a temporary directory.

    Parameters
    ----------
    zip_path : str
        Path to the .zip / .psz file.
    internal_paths : Iterable[str]
        Paths inside the ZIP to extract.

    Returns
    -------
    temp_dir : str
        Path to the temporary directory.
    extracted_files : dict
        Mapping {internal_path -> extracted_disk_path}

    NOTE:
    - Caller is responsible for keeping temp_dir alive.
    - All files are flattened into the same directory.
    """
    temp_dir = tempfile.mkdtemp()
    extracted = {}

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_files = set(zip_ref.namelist())

        for internal_path in internal_paths:
            if internal_path not in zip_files:
                raise FileNotFoundError(
                    f"'{internal_path}' not found in ZIP '{zip_path}'"
                )

            filename = os.path.basename(internal_path)
            dst_path = os.path.join(temp_dir, filename)

            with zip_ref.open(internal_path) as src, open(dst_path, "wb") as dst:
                dst.write(src.read())

            extracted[internal_path] = dst_path

    return temp_dir, extracted