import pprint

import tensorflow as tf
import numpy as np

from dataprovider import *
from memN2N import MemN2N

flags = tf.app.flags
flags.DEFINE_string("dataDir", "../memn2n/data/tasks_1-20_v1-2/en", "Directory of data bAbI")
flags.DEFINE_integer("memory_size", 50, "memory size")
flags.DEFINE_integer("description_length", 11, "max length of description/question")
flags.DEFINE_integer("emb_size", 20, "embedding_size")
flags.DEFINE_integer("vocab_size", -1, "specify size of vocabulary")
flags.DEFINE_integer("batch_size", 32, "training batch size")
flags.DEFINE_float("learning_rate", 0.01, "training learning rate")
flags.DEFINE_integer("num_epoch", 10, "training epochs")

Flags = flags.FLAGS
pp = pprint.PrettyPrinter()
pp.pprint(Flags.__flags)

# define data
data = GenData('data/tasks_1-20_v1-2/en', memory_size = Flags.memory_size, description_size = Flags.description_length, vocab_size = Flags.vocab_size, taskid = 8)
Flags.vocab_size = len(data.vocab)

# define model
sess = tf.Session()
memnet = MemN2N(Flags, sess)
memnet.train(data)






