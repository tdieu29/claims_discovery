"""
Entry module
"""

import csv
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import requests

from .execute import Execute

sys.path.insert(1, Path(__file__).parent.parent.parent.absolute().__str__())

from config.config import logger  # noqa: E402

# Define remote URL and temporary directory
URL = "https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com"
DIRECTORY = os.path.join(tempfile.gettempdir(), "metadata")


class Entry:
    """
    Transforms a list of metadata files into an entry-dates.csv file.
    """

    @staticmethod
    def download(maxdate):
        """
        Downloads the latest metadata file along with the monthly versions. This
        method can filter to a view of entry dates for a specific day by setting the
        maxdate parameter.

        Args:
            maxdate: limits the max date to pull, if None pulls latest, can be used to pull
                     entry dates at the time of a specific file.
        """

        # Create output directory
        if not os.path.exists(DIRECTORY):
            os.mkdir(DIRECTORY)

        # Read list of dates from AI2 CORD-19 page
        changelog = requests.get("%s/latest/changelog" % URL)
        dates = [
            line
            for line in changelog.text.splitlines()
            if re.match(r"\d{4}\-\d{2}\-\d{2}", line)
        ]

        # Sort dates (ascending)
        dates = sorted(dates)

        # Filter by max date
        if maxdate:
            maxdate = datetime.strptime(maxdate, "%Y-%m-%d")
            dates = [d for d in dates if datetime.strptime(d, "%Y-%m-%d" <= maxdate)]

        # Reduce files down to monthly (except latest file)
        for date in dates:
            # Current date
            current = datetime.strptime(date, "%Y-%m-%d")
            if current.day == 1 or date == dates[-1]:
                url = f"{URL}/{date}/metadata.csv"
                path = os.path.join(DIRECTORY, "%s.csv" % date)
                logger.info(f"Retrieving {url} to {path}")

                # Only pull  file if it's not already cached
                if not os.path.exists(path):
                    urlretrieve(url, path)

    @staticmethod
    def run(output=None, maxdate=None):
        """
        Builds an entry-dates.csv file.

        Args:
            output ([type]): output directory, defaults to current working directory
            maxdate ([type]): limits the max date to pull, if None pulls latest, can be used to pull
                                        entry dates at the time of a specific file.
        Returns:
            dataframe with entry dates
        """

        # Download metadata files
        Entry.download(maxdate)

        # Get sorted list of metadata csv files
        files = sorted(
            f
            for f in os.listdir(DIRECTORY)
            if os.path.isfile(os.path.join(DIRECTORY, f))
            and re.match(r"\d{4}-\d{2}-\d{2}\.csv", f)
        )

        uids = {}

        # Process each file, first time id is seen is considered entry date
        for metadata in files:
            # Parse date from filename
            date = os.path.splitext(metadata)[0]
            with open(os.path.join(DIRECTORY, metadata), encoding="utf8") as csvfile:
                for row in csv.DictReader(csvfile):
                    # Get hash value
                    sha = Execute.getHash(row)

                    # Update if hash not seen or cord uid has changed
                    if sha not in uids or row["cord_uid"] != uids[sha][0]:
                        uids[sha] = (row["cord_uid"], sha, date)

        # Create output directory if necessary
        if output:
            os.makedirs(output, exist_ok=True)

        # Output file
        output = (
            os.path.join(output, "entry-dates.csv") if output else "entry-dates.csv"
        )

        # Build dataframe
        df = pd.DataFrame(uids.values(), columns=["cord_uid", "sha", "date"])
        df.to_csv(output, index=False)

        return df


if __name__ == "__main__":
    Entry.run(
        sys.argv[1] if len(sys.argv) > 1 else None,
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
