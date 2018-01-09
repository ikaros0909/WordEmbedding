'''
Created on 2017. 5. 22.

@author: danny
'''
import pymssql

def db_connect_mssql():
    con = pymssql.connect(user='id',password='pw',host='DANNY\SQLEXPRESS',database='db', port='1433')
    #con = db_connect_mssql()
    return con


# cur=con.cursor()    
# print ("Opened database successfully")
# print ("Table created successfully")
# cur.close()
 
def db_connect_oracle(): 
    import os
    #os.environ["NLS_LANG"]=".AL32UF8"
    # os.environ["NLS_LANG"]=".KO16MSWIN949"
    # os.environ["NLS_LANG"]="AMERICAN_AMERICA.KO16MSWIN949"
    os.environ["NLS_LANG"]="KOREAN_KOREA.KO16MSWIN949"
    os.chdir("E:\OracleClient")
     
    import cx_Oracle
    con = cx_Oracle.connect('id/pw@ip/db')
    return con
 
#     print (con.version)
#       
#     for record in db:
#         print(record)
#        
#     con.close()
