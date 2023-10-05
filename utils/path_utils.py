from pathlib import Path
import os
from typing import Union, Dict, Optional, List


def get_project_root() -> Path:
    """Return the root directory of the TLDR project."""
    current_file_path = os.path.abspath('.')
    while os.path.basename(current_file_path) != 'TLDR':
        current_file_path = os.path.dirname(current_file_path)
        if os.path.basename(current_file_path) == os.path.dirname(current_file_path):
            # We've reached the top-level directory without finding 'TLDR'
            raise ValueError("'TLDR' directory not found in the path hierarchy.")
    return current_file_path


def ensure_directory_exists(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def file_exists(filepath: Path) -> bool:
    return filepath.is_file()


def get_data_directory() -> Path:
    return get_project_root() / "datasets"


def get_directory_structure(rootdir: Union[str, Path],
                            ignore_hidden: bool = True,
                            ignore_gitignored: bool = True,
                            ignore_extensions: Optional[List[str]] = None) -> Dict[
    str, Union[None, 'Dict[str, Union[None, dict]]']]:
    """
    Generates a nested dictionary that represents the folder structure of rootdir.

    Parameters:
    - rootdir (Union[str, Path]): The root directory whose structure needs to be fetched. Can be a string or a Path object.
    - ignore_hidden (bool): If True, will ignore files and directories starting with a dot.
    - ignore_gitignored (bool): If True, will ignore files listed in .gitignore.
    - ignore_extensions (List[str], optional): List of file extensions to ignore. e.g. ['.pyc', '.log']

    Returns:
    - Dict[str, Union[None, Dict]]: A nested dictionary representing the directory structure.
      Files are represented with None as their value.
    """
    rootdir = Path(rootdir)  # Ensure rootdir is a Path object
    dir_structure = {}
    git_ignored_files = set()

    # Read .gitignore and collect the list of ignored files.
    if ignore_gitignored:
        gitignore_path = rootdir / ".gitignore"
        if gitignore_path.exists():
            with gitignore_path.open() as f:
                for line in f:
                    # Remove comments and strip whitespaces
                    line = line.split("#")[0].strip()
                    if line:  # If it's not an empty line
                        git_ignored_files.add(rootdir / line)

    for dirpath, dirnames, filenames in os.walk(rootdir):
        # Filter out hidden directories and files
        if ignore_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            filenames[:] = [f for f in filenames if not f.startswith(".")]

        # Filter out gitignored files and directories
        if ignore_gitignored:
            dirnames[:] = [d for d in dirnames if rootdir / dirpath / d not in git_ignored_files]
            filenames[:] = [f for f in filenames if rootdir / dirpath / f not in git_ignored_files]

        # Filter out files with specified extensions
        if ignore_extensions:
            filenames[:] = [f for f in filenames if not f.endswith(tuple(ignore_extensions))]

        # Process directories and files to create the structure
        subpath = dirpath[len(str(rootdir)):].lstrip(os.path.sep)
        subdir = dir_structure

        for dirname in subpath.split(os.path.sep):
            if not dirname:  # root case
                continue
            subdir = subdir.setdefault(dirname, {})

        for filename in filenames:
            subdir[filename] = None

    return dir_structure


def merge_paths(path1: Union[str, Path], path2: Union[str, Path]) -> Path:
    """
    Merges two paths.

    Parameters:
    - path1 (Union[str, Path]): The first path.
    - path2 (Union[str, Path]): The second path to be appended to the first.

    Returns:
    - Path: The merged path.
    """
    return Path(path1) / Path(path2)

