import shutil

from os.path import exists

def clean_archive_output(archive_folders):
    for archive_folder in archive_folders:
        if exists(archive_folder):
            shutil.rmtree(archive_folder)
