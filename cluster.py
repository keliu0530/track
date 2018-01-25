#!/usr/bin/python
import MySQLdb
import pandas as pd
import numpy as np
    
db = MySQLdb.connect(host="keliudb.corzaau5yusv.us-east-2.rds.amazonaws.com",    # your host, usually localhost
                     user="keliu",         # your username
                     passwd="19930530",  # your password
                     db="PuertoRico")        # name of the data base

temp = pd.read_sql("SELECT safegraph_id, x, y FROM PhoneData WHERE safegraph_id=" + "'"+ 'a079a7eda741f073fdbabc05d0357c5f9c2293465514d536fa30f098a49e1e32' +"'", con=db)
