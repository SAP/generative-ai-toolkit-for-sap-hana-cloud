"""
Utility functions for the HANA ML tools.
"""
import os
import shutil
import json
from pathlib import Path
import logging
from datetime import datetime, date
from pandas import Timestamp
from numpy import int64

#pylint: disable=too-many-nested-blocks, unexpected-keyword-arg, invalid-name

logger = logging.getLogger(__name__)

def convert_cap_to_hdi(source_dir, target_dir, archive=True):
    """
    Convert a CAP project structure to an HDI structure.
    Parameters
    ----------
    source_dir : str
        The source directory containing the CAP project files.
    target_dir : str
        The target directory where the HDI structure will be created.
    archive : bool, optional
        If True, the function will create an archive of the source directory.
        Default is True.
    """
    target_path = Path(target_dir)
    if target_path.exists() and target_path.is_dir():
        if any(target_path.iterdir()):
            if archive:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                archive_path = f"archive_{target_dir}_{timestamp}.tar.gz"
                shutil.make_archive(archive_path, 'gztar', target_dir)
                # delete the target directory after archiving including subdirectories except the archive
                for item in target_path.iterdir():
                    if item.name != f"{target_dir}.tar.gz":
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                logger.info("Created archive: %s", archive_path)
            else:
                logger.info("Target directory %s already exists and is not empty.", target_dir)
                raise FileExistsError(f"The target_dir {target_dir} is not empty. Please provide an empty directory.")
    db_src = os.path.join(Path(target_dir), "db", "src")
    db_cfg = os.path.join(Path(target_dir), "db", "cfg")
    srv_dir = os.path.join(Path(target_dir), "srv")
    os.makedirs(db_src, exist_ok=True)
    os.makedirs(db_cfg, exist_ok=True)
    os.makedirs(srv_dir, exist_ok=True)
    cap_db = Path(os.path.join(Path(source_dir), "db"))
    src_files = Path(os.path.join(cap_db, "src")).glob("*")
    for file in src_files:
        if file.suffix == ".cds":
            target_file = os.path.join(db_src, f"{file.stem}.hdbcds")
            shutil.copy2(file, target_file)
        else:
            shutil.copy2(file, os.path.join(db_src, file.name))
    for cds_file in cap_db.glob("*.cds"):
        target_file = os.path.join(db_src, f"{cds_file.stem}.hdbcds")
        shutil.copy2(cds_file, target_file)
    srv_source = Path(os.path.join(Path(source_dir), "srv"))
    if srv_source.exists():
        shutil.copytree(srv_source, srv_dir, dirs_exist_ok=True)
    hdi_config = os.path.join(db_cfg, ".hdiconfig")
    with open(hdi_config, "w") as f:
        json.dump({
            "file": {
                "path": os.path.join("db", "src"),
                "build_plugins": [
                    {"plugin": "com.sap.hana.di.cds"},
                    {"plugin": "com.sap.hana.di.procedure"},
                    {"plugin": "com.sap.hana.di.synonym"},
                    {"plugin": "com.sap.hana.di.grant"}
                ]
            }
        }, f, indent=2)

class _CustomEncoder(json.JSONEncoder):
    """
    This class is used to encode the model attributes into JSON string.
    """
    def default(self, obj): #pylint: disable=arguments-renamed
        if isinstance(obj, (Timestamp, datetime, date)):
            # Convert Timestamp, datetime or date to ISO string
            return obj.isoformat()
        elif isinstance(obj, (int64, int)):
            # Convert numpy int64 or Python int to Python int
            return int(obj)
        # Let other types use the default handler
        return super().default(obj)


def add_stopping_hint(x : str):
    """Added the hint for stopping the execution when an error message is returned."""
    return (x + ". Please stop the execution and return.").replace("..", ".")
