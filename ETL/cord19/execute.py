"""
Transforms and loads CORD-19 data into an articles database.
"""

import csv
import hashlib
import os.path
from multiprocessing import Pool

from config.config import logger

from ..article import Article
from ..sqlite import SQLite
from .section import Section
from .sentence import Sentence


class Execute:
    """
    Transforms and loads CORD-19 data into an articles database.
    """

    @staticmethod
    def getHash(row):
        """
        Gets sha hash for this row. Builds one from the title if no body content is available.

        Args:
            row ([type]): input row

        Returns:
            sha1 hash id
        """

        # Use sha1 provided, if available
        sha = row["sha"].split("; ")[0] if row["sha"] else None
        if not sha:
            # Fallback to sha1 of title
            sha = hashlib.sha1(row["title"].encode("utf-8")).hexdigest()

        return sha

    @staticmethod
    def getUrl(row):
        """
        Parses the url from the input row.

        Args:
            row ([type]): input row

        Returns:
            url
        """

        if row["url"]:
            # Filter out API reference links
            urls = [url for url in row["url"].split("; ") if "https://api." not in url]
            if urls:
                return urls[0]

        # Default to DOI
        return "https://doi.org/" + row["doi"]

    @staticmethod
    def entryDates(indir, entryfile):
        """
        Loads an entry date lookup file into memory.

        Args:
            indir: input directory
            entryfile: path to entry dates file

        Returns:
            dictionary of cord uid -> entry date
        """

        # sha - (cord uid, date) mapping
        entries = {}

        # Default path to entry files if not provi
        if not entryfile:
            entryfile = os.path.join(indir, "entry-dates.csv")

        # Load in memory date lookup
        with open(entryfile) as csvfile:
            for row in csv.DictReader(csvfile):
                entries[row["sha"]] = (row["cord_uid"], row["date"])

        # Reduce down to entries only in metadata
        dates = {}
        count = 0
        with open(os.path.join(indir, "metadata.csv"), encoding="utf8") as csvfile:
            for row in csv.DictReader(csvfile):
                # Lookup hash
                sha = Execute.getHash(row)

                # Lookup record
                uid, date = entries[sha]

                # Store date if cord uid maps to value in entries
                if row["cord_uid"] == uid:
                    dates[uid] = date

                count += 1

        logger.info(f"len(dates): {len(dates)}")
        logger.info(f"count: {count}")
        return dates

    @staticmethod
    def stream(indir, dates, merge):
        """
        Generator that yields rows from a metadata.csv file. The directory is also included.

        Args:
            indir ([type]): input directory
            models ([type]): models directory
            dates ([type]): list of uid - entry dates for current metadata file
            merge ([type]): only merges/processes this list of uids, if enabled
        """
        # Filter out duplicate ids
        ids, hashes = set(), set()

        with open(os.path.join(indir, "metadata.csv"), encoding="utf8") as csvFile:
            for row in csv.DictReader(csvFile):
                # Cord uid
                uid = row["cord_uid"]

                # sha hash
                sha = Execute.getHash(row)

                # Only process if all conditions below met:
                # - Merge set to None (must check for None as merge can be an empty set) or uid in list of ids to merge
                #  - cord uid in entry date mapping
                #  - cord uid and sha hash not already processed
                if (
                    (merge is None or uid in merge)
                    and uid in dates
                    and uid not in ids
                    and sha not in hashes
                ):
                    yield (row)

                # Add uid and sha as processed
                ids.add(uid)
                hashes.add(sha)

        return None

    @staticmethod
    def process(row):
        """
        Processes a single row

        Args:
            params (tuple?): (row, indir)

        Returns:
            article
        """

        # Get text spans in each document (title + abstract)
        sections = Section.parse(row)
        if sections == []:
            return "No title and no abstract"

        # List of sentences in each abstract
        sentences = Sentence.parse(row)

        # Article metadata - id, source, title, abstract, published date, authors, publication, reference
        metadata = (
            row["cord_uid"],
            row["source_x"],
            row["title"],
            row["abstract"],
            row["publish_time"],
            row["authors"],
            row["journal"],
            Execute.getUrl(row),
        )

        return Article(metadata, sections, sentences)

    @staticmethod
    def run(indir, outdir, entryfile, merge_url):
        """
        Main execution method.

        Args:
            indir ([type]): input directory
            outdir ([type]): output directory
            entryfile ([type]): path to entry dates file
            merge_url ([type]): database url to use for merging prior results
        """

        logger.info(f"Building articles database from {indir}")

        # Create database
        db = SQLite(outdir)  # outdir = cord19_data/database

        # Load entry dates
        dates = Execute.entryDates(indir, entryfile)  # dates[uid] = entry_date

        # Merge existing db, if present
        if merge_url:
            merge_uids = db.merge(merge_url, dates)
            logger.info(
                "Merged results from existing articles database. "
                f"len(merge_uids): {len(merge_uids)}"
            )
        else:
            merge_uids = None

        # Create process pool
        with Pool(os.cpu_count()) as pool:
            for article in pool.imap(
                Execute.process, Execute.stream(indir, dates, merge_uids), 100
            ):

                if article == "No title and no abstract":
                    continue  # Skip row with no title and no abstract

                # Get unique id
                uid = article.uid()

                # Append entry date
                article.metadata = article.metadata + (dates[uid],)

                # Save article
                db.save(article)

        # Complete processing
        db.complete()

        # Commit and close
        db.close()
