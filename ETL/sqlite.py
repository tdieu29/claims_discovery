"""
SQLite module
"""

import os 
import sqlite3 

from datetime import datetime, timedelta 

class SQLite(): 
    """
    Defines data structures and methods to store article content in SQLite. 
    """

    # Articles schema 
    ARTICLES = {
        "Article_Id": "TEXT PRIMARY KEY", 
        "Source": "TEXT",
        "Title": "TEXT",
        "Abstract": "TEXT",
        "Published_Date": "DATETIME",
        "Authors": "TEXT", 
        "Journal": "TEXT",
        "Url": "TEXT",
        "Entry_Date": "DATETIME"
    }

    # Sentences schema
    SENTENCES = {
        "Sentence_Id": "INTEGER PRIMARY KEY",
        "Article_Id": "TEXT",
        "Sentence_Index": "INTEGER",
        "Sentence": "TEXT"
    }

    # Sections schema
    SECTIONS = {
        "Section_Id": "INTEGER PRIMARY KEY",
        "Article_Id": "TEXT",
        "Span_Index": "INTEGER", 
        "Text": "TEXT"
    }

    # SQL statements 
    CREATE_TABLE = "CREATE TABLE IF NOT EXISTS {table} ({fields})"
    INSERT_ROW = 'INSERT INTO {table} ({columns}) VALUES ({values})'
    CREATE_INDEX = "CREATE INDEX section_article ON sections (Article_Id, Span_Index)"

    # Merge SQL statements 
    ATTACH_DB = "ATTACH DATABASE '{path}' as {name}"
    DETACH_DB = "DETACH DATABASE '{name}'"
    MAX_ENTRY = "SELECT MAX(Entry_Date) from {name}.articles"
    LOOKUP_ARTICLE = "SELECT Article_Id FROM {name}.articles WHERE Article_Id = ? AND Entry = ?"
    MERGE_ARTICLE = "INSERT INTO articles SELECT * FROM {name}.articles WHERE Article_Id = ?"  
    MERGE_SECTIONS = "INSERT INTO sections SELECT * FROM {name}.sections WHERE Article_Id = ?" 
    UPDATE_ENTRY = "UPDATE articles SET Entry_Date = ? WHERE Article_Id = ?"
    ARTICLE_COUNT = "SELECT COUNT(1) FROM articles"  
    SECTION_COUNT = "SELECT MAX(Section_Id) FROM sections"

    def __init__(self, outdir):
        """
        Creates and initializes a new output SQLite database.

        Args:
            outdir([str?]): output_directory
        """

        # Create if output path does not exist 
        os.makedirs(outdir, exist_ok=True)

        # Output database file 
        dbfile = os.path.join(outdir, "articles.sqlite")

        # Delete existing file ?????
        if os.path.exists(dbfile):
            os.remove(dbfile)
        
        # Index fields
        self.aindex, self.sindex, self.sentIndex = 0, 0, 0

        # Create output database 
        self.db = sqlite3.connect(dbfile)

        # Create database cursor 
        self.cur = self.db.cursor()

        # Create `articles` table 
        self.create(SQLite.ARTICLES, 'articles')

        # Create `sections` table 
        self.create(SQLite.SECTIONS, 'sections')

        # Create `sentences` table
        self.create(SQLite.SENTENCES, 'sentences')

        # Start transaction
        self.cur.execute('BEGIN')
    
    def create(self, schema, name):
        """
        Creates a SQLite table.

        Args:
            schema ([type]): table schema
            name ([type]): table name
        """

        columns = ["{} {}".format(name, dtype) for name, dtype in schema.items()]
        create = SQLite.CREATE_TABLE.format(table=name, fields=", ".join(columns))

        try:
            self.cur.execute(create)
        except Exception as e: 
            print(create)
            print("Failed to create table: " + e)
    
    def insert(self, table, name, row):
        """
        Builds and inserts a row.

        Args:
            table ([dict]): table schema 
            name ([str]): table name
            row ([type]): row to insert 
        """
        # Build insert statement 
        columns = [colName for colName, _ in table.items()]

        #'INSERT INTO {table} ({columns}) VALUES ({values})'
        insert = SQLite.INSERT_ROW.format(table=name,
                                        columns = ", ".join(columns),
                                        values = ("?, " * len(columns))[:-2])
        try: 
            # Execute insert statement 
            self.cur.execute(insert, self.values(table, row, columns))
        except Exception as ex:
            print("Error inserting row: {}".format(row), ex)
    
    def values(self, table, row, columns):
        """
        Formats and converts row into database types based on table schema.
        
        Args:
            table: table schema
            row: row tuple 
            columns: column names

        Returns:
            Database schema formatted row tuple
        """ 

        values = []
        for x, column in enumerate(columns): 
            # Get value 
            value = row[x]

            if table[column].startswith("INTEGER"):
                values.append(int(value) if value else 0)
            elif table[column].startswith("BOOLEAN"):
                values.append(1 if value == "TRUE" else 0)
            elif table[column].startswith("TEXT"):
                # Clean empty text and replace with None 
                values.append(value if value and len(value.strip()) > 0 else None)
            else:
                values.append(value)
        
        return values
    
    def save(self, article): 
        # Article row 
        self.insert(SQLite.ARTICLES, "articles", article.metadata)

        # Increment number of articles processed 
        self.aindex += 1 
        if self.aindex % 1000 == 0: 
            print("Inserted {} articles".format(self.aindex), end="\r")

            # Commit current transaction and start a new one 
            self.transaction()
        
        for (sentence_idx, sentence) in article.sentences:
            # sentence_id, article_id, sentence_index, sentence
            self.insert(SQLite.SENTENCES, "sentences", (self.sentIndex, article.uid(), sentence_idx, sentence))
            self.sentIndex += 1

        for (span_idx, text) in article.sections: 
            # section_id, article_id, span_index, text
            self.insert(SQLite.SECTIONS, "sections", (self.sindex, article.uid(), span_idx, text))
            self.sindex += 1

    def merge(self, url, ids): 
        # List of IDs to set for processing 
        queue = set()

        # Attached database alias
        alias = "merge"

        # Attach database 
        self.db.execute(SQLite.ATTACH_DB.format(path=url, name=alias)) 

        # Only process records newer than 5 days before the last run 
        lastrun = self.cur.execute(SQLite.MAX_ENTRY.format(name=alias)).fetchone()[0]
        print('lastrun1: ', print(lastrun))
        lastrun = datetime.strptime(lastrun, "%Y-%m-%d") - timedelta(days=5)
        print('lastrun2: ', print(lastrun))
        lastrun = lastrun.strftime("%Y-%m-%d")
        print('lastrun3: ', print(lastrun))

        # Search for existing articles 
        for uid, date in ids.items():
            self.cur.execute(SQLite.LOOKUP_ARTICLE.format(name=alias), [uid, date])
            if not self.cur.fetchone() and date > lastrun: 
                # Add uid to process 
                queue.add(uid)
            else:
                # Copy existing record 
                self.cur.execute(SQLite.MERGE_ARTICLE.format(name=alias), [uid]) 
                self.cur.execute(SQLite.MERGE_SECTIONS.format(name=alias), [uid]) 

                # Sync entry date with ids list 
                self.cur.execute(SQLite.UPDATE_ENTRY, [date, uid]) 

        # Set current index positions 
        self.aindex = int(self.cur.execute(SQLite.ARTICLE_COUNT.format(name=alias)).fetchone()[0]) + 1
        self.sindex = int(self.cur.execute(SQLite.SECTION_COUNT.format(name=alias)).fetchone()[0]) + 1

        # Commit transaction
        self.db.commit()

        # Detach database 
        self.db.execute(SQLite.DETACH_DB.format(name=alias))

        # Start new transaction
        self.cur.execute("BEGIN")

        # Return list of new/ updated ids to process 
        return queue

    def transaction(self):
        """
        Commites current transaction and create a new one.
        """
        self.db.commit()
        self.cur.execute("BEGIN")

    def complete(self):
        print("Total articles inserted: {}".format(self.aindex))

        # Create articles index for spans table 
        self.cur.execute(SQLite.CREATE_INDEX)

    def close(self):
        self.db.commit()
        self.db.close()
    
    def execute(self, sql):
        """
        Executes SQL statement against open cursor.
        
        Args:
            sql: SQL statement
        """

        self.cur.execute(sql)
