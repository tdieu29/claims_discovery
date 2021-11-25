"""
Defines main entry point for ETL process.
"""

import os
import shutil
import sys

from .execute import Execute

if __name__ == "__main__":
    if os.path.exist("cord19_data/database/articles.sqlite"):
        os.makedirs("temp", exist_ok=True)
        shutil.copy("cord19_data/database/articles.sqlite", "temp/")

    Execute.run(
        sys.argv[1] if len(sys.argv) > 1 else "cord19_data/metadata",
        sys.argv[2] if len(sys.argv) > 2 else "cord19_data/database",
        sys.argv[3] if len(sys.argv) > 3 else "cord19_data/metadata/entry-dates.csv",
        sys.argv[4] if len(sys.argv) > 4 else "temp/articles.sqlite",
    )
