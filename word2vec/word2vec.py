import tensorflow as tf
import numpy as np
import random, io, sys, math, json
from mysql_hooks import MySQLInterface
from nltk.tokenize import TweetTokenizer as Tokenizer
from _mysql_exceptions import ProgrammingError as MySQLError, OperationalError as MySQLOperationalError
class Word2VecModel: 
    def __init__(self, db, embedding_size=100, num_sampled=16, batch_size=128, learn_rate=1):
        self.db=db
        vocabulary_size=int(db.query("SELECT COUNT(DISTINCT ID) FROM WORDS")[0][0])
        self.vocabulary_size=vocabulary_size
        self.embedding_size=embedding_size
        self.num_sampled=num_sampled
        self.batch_size=batch_size
        self.embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        self.nce_weights = tf.Variable(
                  tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                            stddev=1.0 / math.sqrt(self.embedding_size)))
        self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size,1 ])
        self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
        self.loss=loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights,biases=self.nce_biases, labels=self.train_labels, inputs=self.embed, num_sampled=self.num_sampled, num_classes=self.vocabulary_size))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(self.loss)
        try:    
            try:
                db.execute("CREATE TABLE WORD2VEC (ID INT, WEIGHTS LONGTEXT, BIASES LONGTEXT, EMBEDDINGS LONGTEXT)")
            except MySQLError:
                pass
        except MySQLOperationalError:
            pass
        weights_serialized=db.query("SELECT WEIGHTS, BIASES, EMBEDDINGS FROM WORD2VEC ORDER BY ID DESC LIMIT 1")
        self.W=None
        self.b=None
        self.e=None
        if len(weights_serialized)>0:
            print("found existing network in db...deserializing")
            s_weights=weights_serialized[0][0]
            s_biases=weights_serialized[0][1]
            s_embeddings=weights_serialized[0][2]
            self.W=np.array(json.loads(s_weights))
            self.b=np.array(json.loads(s_biases))
            self.e=np.array(json.loads(s_embeddings))    
    def load(self, session):
        if self.W is not None:#if we have load previous embeddings, weights, biases from the database...
            session.run(tf.assign(self.nce_weights, self.W))
            session.run(tf.assign(self.nce_biases, self.b))
            session.run(tf.assign(self.embeddings, self.e))

    def train(self, session=None,  skip_window=5, query="SELECT TEXT FROM MESSAGES ORDER BY RAND() LIMIT 100"):
        if session is None:
            session=tf.Session()
        batch_labels=[]
        batch_inputs=[]
        #keep looking through tweets until we have a whole training batch
        while len(batch_inputs)<self.batch_size:
                qresult=self.db.query(query)
                for tweet in qresult:
                    tokens=tweet[0].split()
                    if len(tokens)<1:
                        continue
                    for i,tok0 in enumerate(tokens):
                        tok1=tokens[i+min(max(random.choice(list(range(-skip_window,0))+list(range(1,skip_window))),-i),len(tokens)-i-1)]

                        batch_labels.append(int(tok0))
                        batch_inputs.append(int(tok1))
                    if len(batch_inputs)>=self.batch_size:
                        break
        avg_loss=0
        output=[self.optimizer,self.loss,self.nce_weights, self.nce_biases, self.embeddings]
        feed={self.train_inputs: np.array(batch_inputs[:self.batch_size]), self.train_labels: np.array(batch_labels[:self.batch_size]).reshape((self.batch_size,1))}
        batch_inputs=batch_inputs[self.batch_size:]
        batch_labels=batch_labels[self.batch_size:]
        _,cur_loss,self.W,self.b,self.e=session.run(output,feed_dict=feed)
        return cur_loss
    def save(self):
        s_weights=io.BytesIO()
        weights_serialized=json.dumps(self.W.tolist())
        biases_serialized=json.dumps(self.b.tolist())
        embeddings_serialized=json.dumps(self.e.tolist())
        ID=int(self.db.query('SELECT COUNT(ID) FROM WORD2VEC')[0][0])
        self.db.execute('TRUNCATE TABLE WORD2VEC')
        self.db.execute('INSERT INTO WORD2VEC VALUES(%i, "%s", "%s", "%s")'%(ID+1,weights_serialized, biases_serialized, embeddings_serialized))
class Word2Vec:
    def __init__(self, db, sess=None):
        self.db=db 
        serialized=db.query("SELECT EMBEDDINGS FROM WORD2VEC LIMIT 1")
        self.embeddings=tf.Variable(np.array(json.loads(serialized[0][0])))
        self.input=tf.placeholder(tf.int32, shape=[1])
        self.embed=tf.nn.embedding_lookup(self.embeddings,self.input)
        self.session=sess
        if sess is None:
            self.session=tf.session()
    def __call__(self, word):
         session=self.session
         w=self.db.query('SELECT ID FROM WORDS WHERE WORD=%s LIMIT 1', word)[0]
         if len(w)==0:
            return np.zeros(100)
         elif w[0] is None:
            return np.zeros(100)
         index_label=int(w)
         return self.session.run([self.embed], feed_dict={self.input:np.array([index_label])})
                
def word2vec_all(N=10, bsize=128):
    db=MySQLInterface(*sys.argv[1:])
    w2vec=Word2VecModel(db, batch_size=bsize)
    init_op=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init_op)
    w2vec.load(sess)
    for t in range(N):
        print(w2vec.train(sess))
    w2vec.save()
if __name__=='__main__':
    word2vec_all()
