import argparse
import pickle

import tensorflow as tf
import numpy as np


DATA_DIR = 'data/penn-UNK-train.txt' # clean dataset
CKPT_DIR = 'checkpoints/'

RESTORE = False

WINDOW_SIZE 	= 50
BATCH_SIZE 		= 50
EMBEDDING_SIZE 	= 128
RNN_SIZE 		= 256
EPOCH_NUM 		= 20

class LanguageModel(object):
	def __init__(self, vocab_size):
		self.vocab_size = vocab_size

		self.inputs 	= tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])
		self.labels 	= tf.placeholder(tf.int32, shape=[None, WINDOW_SIZE])
		self.input_lens = tf.placeholder(tf.int32, shape=[None])

		self.logits 	= self.forward()
		self.loss 		= self.compute_loss()
		self.optim 		= self.optimizer()
		self.perplexity = self.compute_perplexity()

	def forward(self):
		# embedding
		E = tf.Variable(tf.truncated_normal([self.vocab_size, EMBEDDING_SIZE], stddev=0.01))
		embeddings = tf.nn.embedding_lookup(E, self.inputs) 

		# rnn
		rnn = tf.contrib.rnn.LSTMCell(RNN_SIZE)
		outputs, next_state = tf.nn.dynamic_rnn(rnn, embeddings, dtype=tf.float32)

		# fully connected
		W = tf.Variable(tf.truncated_normal([RNN_SIZE, self.vocab_size], stddev=0.01))
		b = tf.Variable(tf.truncated_normal([self.vocab_size,], stddev=0.01))
		return tf.tensordot(outputs, W, axes=[[2], [0]]) + b # [BATCH_SIZE, WINDOW_SIZE, vocab_size]

	def compute_loss(self):
		return 	tf.contrib.seq2seq.sequence_loss(
					self.logits, 
					self.labels, 
					tf.sequence_mask(self.input_lens, dtype=tf.float32, maxlen=WINDOW_SIZE)
				)

	def optimizer(self):
		return tf.train.AdamOptimizer(1e-3).minimize(self.loss)

	def compute_perplexity(self):
		return tf.exp(self.loss)

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
	preprocessed, sentence_lens = [], []
	for sentence in sentences:
		sentence_len = len(sentence)
		sentence_lens.append(min(WINDOW_SIZE, len(sentence)))
		preprocessed.append([vocab[word] for word in 
			(sentence + ['STOP']*(window_size-sentence_len))[:window_size]]) # pad with STOP
	return np.array(preprocessed), np.array(sentence_lens)

def shuffle(l):
	shuffled_indices = np.random.permutation(len(l[0]))
	res = []
	for x in l:
		res.append(np.array([x[i] for i in shuffled_indices]))
	return res

def train(model, sess, saver, sentences, sentence_lens):
	n_sentences = len(sentences)
	for epoch in range(EPOCH_NUM):
		print('----- epoch %3d -----' % epoch)
		inputs, lens = shuffle([sentences, sentence_lens])
		step = 0

		for start in range(0, n_sentences, BATCH_SIZE):
			end = min(n_sentences, start+BATCH_SIZE)
			l, p, _ = sess.run([model.loss, model.perplexity, model.optim],
						feed_dict={
							model.inputs: 	inputs[start:end, :-1],
							model.labels: 	inputs[start:end, 1:],
							model.input_lens:		lens[start:end]
						})
			if step%100 == 0:
				print('step %4d >> loss: %.3f, perplexity: %.3f' % (step, l, p))
			step += 1

		if (epoch+1) % 5 == 0:
			ckpt_path = CKPT_DIR + 'model_epoch{}.ckpt'.format(epoch)
			save_path = saver.save(sess, CKPT_DIR + 'model_epoch{}.ckpt'.format(epoch))
			print('Checkpoint saved to', save_path)

def gen_text(model, sess, start_word, vocab, inverse_vocab):
	curr_sentence = []
	next_word = vocab[start_word]
	for i in range(WINDOW_SIZE):
		curr_sentence = curr_sentence + [next_word]
		padded = np.array(curr_sentence + [vocab['STOP']] * (WINDOW_SIZE-i-1))
		logits = sess.run(model.logits, feed_dict={model.inputs: padded.reshape(1, -1)})
		next_word = np.argmax(logits[0][i])
		if next_word == vocab['STOP']: break
	to_words = [inverse_vocab[w] for w in curr_sentence]
	return to_words

def main():


	if RESTORE:

		# vocab
		with open(CKPT_DIR + 'vocab.pkl', 'rb') as f:
			vocab, inverse_vocab = pickle.load(f)
		vocab_size = len(vocab)

		# model
		model = LanguageModel(vocab_size)
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
		sentences, sentence_lens = tokenize_pad(sentences, WINDOW_SIZE+1, vocab)

		# train the language model
		model = LanguageModel(vocab_size)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		train(model, sess, saver, sentences, sentence_lens)

	# generate sentences
	start_word = 'The'
	print('Start word = "%s"' % start_word)
	out = gen_text(model, sess, start_word, vocab, inverse_vocab)
	print('Output:', out)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--restore", action='store_true')
	args = parser.parse_args()
	RESTORE = args.restore

	main()