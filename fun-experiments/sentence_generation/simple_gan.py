import argparse
import pickle

import tensorflow as tf
import numpy as np


DATA_DIR = 'data/penn-UNK-train.txt' # clean dataset
CKPT_DIR = 'checkpoints_gan/'

RESTORE = False

WINDOW_SIZE 	= 30
BATCH_SIZE 		= 50
Z_DIM 			= 64 # random vector dim
RNN_SIZE 		= 512
EPOCH_NUM 		= 20
GEN_TRAIN_NUM 	= 3

class GAN(object):
	def __init__(self, vocab_size):
		self.vocab_size = vocab_size

		self.g_inputs		= tf.placeholder(tf.float32, shape=[None, Z_DIM])
		self.reals 			= tf.placeholder(tf.float32, shape=[None, WINDOW_SIZE, vocab_size]) # real sentences
		self.learning_rate 	= tf.placeholder(tf.float32, shape=None)

		self.g_outputs	= self.generator(self.g_inputs)
		self.d_real		= self.discriminator(self.reals, None)
		self.d_fake		= self.discriminator(self.g_outputs, True)

		self.g_loss		= self.compute_g_loss()
		self.d_loss		= self.compute_d_loss()
		self.g_optim	= self.g_optimize()
		self.d_optim	= self.d_optimize()

	def generator(self, z):
		with tf.variable_scope('generator'):

			# fake embedding
			W1 = tf.Variable(tf.truncated_normal([Z_DIM, WINDOW_SIZE * self.vocab_size], stddev=0.01))
			b1 = tf.Variable(tf.truncated_normal([WINDOW_SIZE * self.vocab_size,], stddev=0.01))
			embeddings = tf.matmul(z, W1) + b1
			embeddings = tf.reshape(embeddings, (-1, WINDOW_SIZE, self.vocab_size))

			# rnn
			rnn = tf.contrib.rnn.LSTMCell(RNN_SIZE)
			outputs, next_state = tf.nn.dynamic_rnn(rnn, embeddings, dtype=tf.float32)

			# fully connected
			W2 = tf.Variable(tf.truncated_normal([RNN_SIZE, self.vocab_size], stddev=0.01))
			b2 = tf.Variable(tf.truncated_normal([self.vocab_size,], stddev=0.01))
			logits = tf.tensordot(outputs, W2, axes=[[2], [0]]) + b2 # [BATCH_SIZE, WINDOW_SIZE, vocab_size]
			return tf.nn.softmax(logits, axis=2)

	def discriminator(self, x, reuse):
		with tf.variable_scope('discriminator', reuse=reuse):

			# reduce 3rd dim
			hidden_size = 64
			W1 = tf.Variable(tf.truncated_normal([self.vocab_size, hidden_size], stddev=0.01))
			b1 = tf.Variable(tf.truncated_normal([hidden_size,], stddev=0.01))
			y = tf.tensordot(x, W1, axes=[[2], [0]]) + b1 # [BATCH_SIZE, WINDOW_SIZE, hidden_size]

			# score
			W2 = tf.Variable(tf.truncated_normal([WINDOW_SIZE * hidden_size, 1], stddev=0.01))
			b2 = tf.Variable(tf.truncated_normal([1,], stddev=0.01))
			y = tf.reshape(y, (-1, WINDOW_SIZE * hidden_size)) # [BATCH_SIZE, WINDOW_SIZE * hidden_size]
			logits = tf.matmul(y, W2) + b2 # [BATCH_SIZE, 1]

			return tf.nn.sigmoid(logits)

	def _log(self, x):
		return tf.log(tf.maximum(x, 1e-5))

	def compute_g_loss(self):
		return tf.reduce_mean(-self._log(self.d_fake))

	def compute_d_loss(self):
		return tf.reduce_mean(-self._log(self.d_real)-self._log(1-self.d_fake))/2

	def g_optimize(self):
		g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
		return tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss, var_list=g_vars)

	def d_optimize(self):
		d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
		return tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss, var_list=d_vars)


def read_file(filename):
	with open(filename, 'r') as f:
		sentences = [line.split() for line in f.readlines()]
	all_words = list(set([word for sentence in sentences for word in sentence]))
	vocab, inverse_vocab = {}, {}
	for i, word in enumerate(all_words):
		vocab[word] = i
		inverse_vocab[i] = word

	# save vocab
	with open(CKPT_DIR + 'vocab.pkl', 'wb') as f:
		pickle.dump((vocab, inverse_vocab), f)

	return sentences, vocab, inverse_vocab

def tokenize_pad(sentences, window_size, vocab):
	preprocessed = []
	for sentence in sentences:
		preprocessed.append([vocab[word] for word in 
			(sentence + ['STOP']*(window_size-len(sentence)))[:window_size]]) # pad with STOP
	return np.array(preprocessed)

def shuffle(l):
	shuffled_indices = np.random.permutation(len(l[0]))
	res = []
	for x in l:
		res.append(np.array([x[i] for i in shuffled_indices]))
	return res

def onehot(sentences, vocab_size):
	vectors = np.zeros([len(sentences), WINDOW_SIZE, vocab_size])
	for i, sentence in enumerate(sentences):
		for w, token in enumerate(sentence):
			vectors[i, w, token] = 1
	return vectors

def train(model, sess, saver, sentences, vocab_size):
	n_sentences = len(sentences)
	for epoch in range(EPOCH_NUM):
		print('----- epoch %3d -----' % epoch)
		inputs = shuffle([sentences])[0]
		step = 0
		for start in range(0, n_sentences, BATCH_SIZE):
			end = min(n_sentences, start+BATCH_SIZE)

			z = np.random.uniform(low=-1, high=1, size=[end-start, Z_DIM])
			reals = onehot(inputs[start:end], vocab_size)

			# train discriminator
			loss_d, _ = sess.run([model.d_loss, model.d_optim], 
									feed_dict = {
										model.g_inputs:			z,
										model.reals: 			reals,
										model.learning_rate:	1e-4	
									})

			# train generator
			for i in range(GEN_TRAIN_NUM):
				loss_g, _ = sess.run([model.g_loss, model.g_optim], 
										feed_dict = {
											model.g_inputs: 		z,
											model.learning_rate:	1e-2})
				# print('  g:', loss_g)

			if step%100 == 0:
				print('step %4d >> g_loss: %.3f, d_loss: %.3f' % (step, loss_g, loss_d))
			step += 1

		if (epoch+1) % 5 == 0:
			ckpt_path = CKPT_DIR + 'model_epoch{}.ckpt'.format(epoch)
			save_path = saver.save(sess, CKPT_DIR + 'model_epoch{}.ckpt'.format(epoch))
			print('Checkpoint saved to', save_path)

# def gen_text(model, sess, start_word, vocab, inverse_vocab):
# 	curr_sentence = []
# 	next_word = vocab[start_word]
# 	for i in range(WINDOW_SIZE):
# 		curr_sentence = curr_sentence + [next_word]
# 		padded = np.array(curr_sentence + [vocab['STOP']] * (WINDOW_SIZE-i-1))
# 		logits = sess.run(model.logits, feed_dict={model.inputs: padded.reshape(1, -1)})
# 		next_word = np.argmax(logits[0][i])
# 		if next_word == vocab['STOP']: break
# 	to_words = [inverse_vocab[w] for w in curr_sentence]
# 	return to_words

def main():

	if RESTORE:

		# vocab
		with open(CKPT_DIR + 'vocab.pkl', 'rb') as f:
			vocab, inverse_vocab = pickle.load(f)
		vocab_size = len(vocab)

		# model
		model = GAN(vocab_size)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(sess, CKPT_DIR + 'model_epoch19.ckpt')
		
	else:

		# read file
		sentences, vocab, inverse_vocab = read_file(DATA_DIR)
		n_sentences = len(sentences)
		vocab_size = len(vocab)
		print('Number of sentences:', n_sentences)
		print('Vocab size:', vocab_size)

		# preprocessing
		sentences = tokenize_pad(sentences, WINDOW_SIZE, vocab)

		# train the language model
		model = GAN(vocab_size)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		train(model, sess, saver, sentences, vocab_size)

	# generate sentences
	# start_word = 'The'
	# print('Start word = "%s"' % start_word)
	# out = gen_text(model, sess, start_word, vocab, inverse_vocab)
	# print('Output:', out)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--restore", action='store_true')
	args = parser.parse_args()
	RESTORE = args.restore

	main()