#import pandas as pd 
#df = pd.read_csv('cord19/metadata/entry-dates.csv')
#print(len(df)) # 753,608

#import csv

#count = 0
#with open('cord19/metadata/entry-dates.csv', mode='r') as csvfile: 
#        for row in csv.DictReader(csvfile): 
#                if row["date"] == "2021-09-06":
#                        count += 1
#print(count)     

##########################################################################################################

# Step 1: python -m ETL.cord19.entry cord19_data/metadata
# Step 2: python -m ETL.cord19 cord19_data/metadata cord19_data/database cord19_data/metadata/entry-dates.csv
#indir = cord19_data/metadata
#outdir = cord19_data/database
#entryfile = cord19_data/metadata/entry-dates.csv
# C:\Users\trang\AppData\Local\Temp\metadata\

##############################################################################################################
import sqlite3 

def show_all():
    db = sqlite3.connect('cord19_data/database/articles.sqlite')

    cur = db.cursor()
    
    #article_id = cur.execute("SELECT Article_Id FROM sections WHERE Section_Id = ?", (1,)).fetchone()[0]
    #print(article_id, type(article_id))

    #article = cur.execute("SELECT * FROM articles WHERE Article_Id = (?)", ('2g4o1alu',)).fetchall()
    #print(article)
        
    #for line in cur.execute("SELECT * FROM sections"):
    #    print(line)
    #    break
    
    #sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
    #tables = cur.execute(sql_query).fetchall()
    
    #print(tables)

    noTitle, noAbstract, noPDate, noAuthors, noJournal, noURL = [], [], [], [], [], []

    for line in cur.execute("SELECT * FROM articles"):
        id, source, title, abstract, p_date, authors, journal, url, e_date = line
        if title == '' or title == None:
            noTitle.append(id)
        if p_date == '' or p_date == None:
            noPDate.append(id)
        if abstract == '' or abstract == None:
            noAbstract.append(id)
        if authors == '' or authors == None:
            noAuthors.append(id)
        if journal == '' or journal == None:
            noJournal.append(id)
        if url == '' or url == None:
            noURL.append(id)

    db.close()

    #print('No Title')
    #print(len(noTitle)) # 0

    #print('No Abstract')
    #print(len(noAbstract)) # 156051

    print('No published date')
    print(len(noPDate))

    print('No authors')
    print(len(noAuthors))

    print('No journal')
    print(len(noJournal))

    #print('noURL')
    #print(len(noURL)) # 0

    #return noTitle, noAbstract, noPDate, noAuthors, noJournal, noURL

show_all()

######################################################################################################################
#import os
#import shutil

#from .execute import Execute as Etl

#print('\ncurrentDirectory: ', os.getcwd()) => C:\Users\trang\claims_discovery

# Copy previous articles database locally for predictable performance
#os.makedirs('temp', exist_ok=True)
#shutil.copy("cord19_data/database/articles.sqlite", "temp/")

#indir = cord19_data/metadata
#outdir = cord19_data/database
#entryfile = cord19_data/metadata/entry-dates.csv
# C:\Users\trang\AppData\Local\Temp\metadata\

# Build SQLite database for metadata.csv and json full text files
#Etl.run(indir="cord19_data/metadata", 
#        outdir="cord19_data/database",
#        entryfile="cord19_data/metadata/entry-dates.csv", 
#        merge_url="temp/articles.sqlite")

##################################################################################################33

#09/06/2021
#Building articles database from cord19\metadata
#len(dates):  653530
#count:  751943
#Total articles inserted: 638109

#10/18/2021
#Building articles database from cord19_data/metadata
#len(dates):  685791
#count:  785268
#Total articles inserted: 669973

#10/25/2021
#len(dates):  701892
#count:  801822
#Total articles inserted: 686096
########################################################################################################