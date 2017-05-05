import poplib, getpass, email
from html2text import html2text
from nltk.tokenize import TweetTokenizer
from mysql_hooks import MySQLInterface
def retrieve_all_messages(username, password, host, port='995'):
    mailbox=poplib.POP3_SSL(host,port)
    mailbox.user(username)
    mailbox.pass_(password)
    message_count=len(mailbox.list()[1])
    for msg_index in range(message_count):
        for msg in mailbox.retr(msg_index+1)[1]:
            yield msg

def retrieve_every_message_body(*args):
    for raw_msg in retrieve_all_messages(*args):
        msg=email.message_from_string(raw_msg.decode('utf-8'))
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                yield part.get_payload()
            elif part.get_content_type() == 'text/html':
                yield html2text(part.get_payload())
def tokenize_every_message_body(db, *args):
    db.execute('CREATE TABLE WORDS (ID INT, WORD VARCHAR(4096))')
    db.execute('CREATE TABLE MESSAGES (ID INT, TEXT LONGTEXT)')
    tknzr=TweetTokenizer()
    for txt in retrieve_every_message_body(*args):
        tokens=[]
        for token in tknzr.tokenize(txt):
            word_ids=db.query_one('SELECT ID FROM WORDS WHERE WORD=%s',token)
            word_id=-1
            if word_ids is not None:
                if len(word_ids)>0:
                    word_id=int(word_ids[0])
            if word_id<0:
                word_id=int(db.query_one('SELECT COUNT(DISTINCT ID) FROM WORDS')[0])
                db.execute('INSERT INTO WORDS VALUES(%i, %%s)'%word_id, token)
            tokens.append(str(word_id))
        i=int(db.query_one('SELECT COUNT(DISTINCT ID) FROM MESSAGES')[0])
        db.execute('INSERT INTO MESSAGES VALUES(%i, %%s)'%i, ' '.join(tokens))

if __name__=='__main__':
    import sys
    tokenize_every_message_body(MySQLInterface(*sys.argv[1:5]),*sys.argv[5:])

