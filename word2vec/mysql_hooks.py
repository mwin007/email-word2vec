import MySQLdb as msql
from _mysql_exceptions import ProgrammingError as MySQLError
class MySQLInterface:
    def __init__(self, hostname, username, pwd, dbname):
        '''connect to a remote mysql database'''
        self.db_connection=msql.connect(host=hostname, user=username, passwd=pwd, db=dbname)
        self.db_cursor=self.db_connection.cursor()
    def query_one(self, query, *args):
        '''get stuff from the sql database'''
        self.db_cursor.execute(query,args)
        return self.db_cursor.fetchone()
    def query(self,query, *args):
        '''get stuff from the sql database'''
        self.db_cursor.execute(query,args)
        return self.db_cursor.fetchall()
    def iter_query(self,query, *args):
        '''get stuff from the sql database'''
        self.db_cursor.execute(query,args)
        while True:
            res_next=self.db_cursor.fetchone()
            if res_next is not None:
                yield res_next
            else:
                return
    def execute(self,command,*args):
        '''push commands to the sql database without getting stuff'''
        self.db_cursor.execute(command, args)
        self.db_connection.commit()
