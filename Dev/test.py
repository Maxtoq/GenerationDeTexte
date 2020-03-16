import os
import re
import random


def generate_ngrams(sentences):
	unigr = {}
	bigr = {}
	trigr = {}
	fourgr = {}
	fivegr = {}
	sixgr = {}
	i = 0
	while i < len(sentences):
		if len(sentences[i]) < 25:
			sentences.pop(i)
			continue
			
		sentences[i] = '>' + sentences[i] + '.<'
		
		for j in range(len(sentences[i])):
			c = sentences[i][j].lower()
			
			if c in unigr:
				unigr[c] += 1
			else:
				unigr[c] = 1
			
			if j >= 1:
				cc = (c_1, c)
				if cc in bigr:
					bigr[cc] += 1
				else:
					bigr[cc] = 1
			
			if j >= 2:
				ccc = (c_2, c_1, c)
				if ccc in trigr:
					trigr[ccc] += 1
				else:
					trigr[ccc] = 1

			if j >= 3:
				cccc = (c_3, c_2, c_1, c)
				if cccc in fourgr:
					fourgr[cccc] += 1
				else:
					fourgr[cccc] = 1

			if j >= 4:
				ccccc = (c_4, c_3, c_2, c_1, c)
				if ccccc in fivegr:
					fivegr[ccccc] += 1
				else:
					fivegr[ccccc] = 1

			if j >= 5:
				cccccc = (c_5, c_4, c_3, c_2, c_1, c)
				if cccccc in sixgr:
					sixgr[cccccc] += 1
				else:
					sixgr[cccccc] = 1
			
			if j >= 4:
				c_5 = c_4
			if j >= 3:
				c_4 = c_3
			if j >= 2:
				c_3 = c_2
			if j >= 1:
				c_2 = c_1
			c_1 = c
		
		i += 1
	
	return sentences, unigr, bigr, trigr, fourgr, fivegr, sixgr

def compute_probs(ngrams, n_1grams=None, lastn_1_char=None):
	probs = {}

	if n_1grams is None:
		# Compute total count of unigram
		tot_unigr = 0
		for k, count in ngrams.items():
			tot_unigr += count
		
		# Compute unigram probs
		for k, count in ngrams.items():
			probs[k] = count / tot_unigr
	else:
		# Compute ngram probs
		for k, count in ngrams.items():
			nope = False
			for k_char, last_char in zip(k, lastn_1_char):
				if k_char != last_char:
					nope = True
			if nope:
				continue
			if len(k) == 2:
				n_1k = k[0]
			else:
				n_1k = k[:-1]
			probs[k] = count / n_1grams[n_1k]

	return probs

def sample_from_dist(dist):
	# 3.6 !!!!!!
	#key = random.choice(dist.keys(), p=dist.values())
	r = random.random()
	prob_sum = 0
	for key, prob in dist.items():
		prob_sum += prob
		if prob_sum >= r:
			break
	return str(key[-1])

def generate_sentence(unigr, bigr=None, trigr=None, fourgr=None, fivegr=None, sixgr=None):
	sentence = '>'
	while True:
		new_char = None
		if sixgr is not None and new_char is None and len(sentence) >= 5:
			# Get 6-gram probs
			sixgr_probs = compute_probs(sixgr, fivegr, sentence[-5:])
			
			if len(sixgr_probs) > 0:
				# Sample from 6-grams
				new_char = sample_from_dist(sixgr_probs)

		if fivegr is not None and new_char is None and len(sentence) >= 4:
			# Get 5-gram probs
			fivegr_probs = compute_probs(fivegr, fourgr, sentence[-4:])
			
			if len(fivegr_probs) > 0:
				# Sample from 5-grams
				new_char = sample_from_dist(fivegr_probs)

		if fourgr is not None and new_char is None and len(sentence) >= 3:
			# Get 4-gram probs
			fourgr_probs = compute_probs(fourgr, trigr, sentence[-3:])
			
			if len(fourgr_probs) > 0:
				# Sample from 4-grams
				new_char = sample_from_dist(fourgr_probs)

		if trigr is not None and new_char is None and len(sentence) >= 2:
			# Get trigram probs
			trigr_probs = compute_probs(trigr, bigr, sentence[-2:])
			
			if len(trigr_probs) > 0:
				# Sample from trigrams
				new_char = sample_from_dist(trigr_probs)
		
		if bigr is not None and new_char is None and len(sentence) >= 1:
			# Get bigram probs
			bigr_probs = compute_probs(bigr, unigr, sentence[-1])
		
			if len(bigr_probs) > 0:
				# Sample from bigrams
				new_char = sample_from_dist(bigr_probs)
		
		if new_char is None:
			# Get unigram probs
			unigr_probs = compute_probs(unigr)

			# Sample from unigrams
			new_char = sample_from_dist(unigr_probs)
		
		sentence += new_char

		# Exit if <END> character generated
		if new_char == '<' or len(sentence) > 1000:
			break

		print(new_char, end='')
	print()
	
if __name__ == '__main__':
	str_file = open('text.txt', 'r').read()
	
	str_file = str_file.replace('\n', ' ')
	sentences = str_file.split('. ')
	
	sentences, unigr, bigr, trigr, fourgr, fivegr, sixgr = generate_ngrams(sentences)
	# Print counts
	#for k in unigr:
	#	print(k, unigr[k], probs_unigr[k])
	#for k in bigr:
	#	print(k, bigr[k], probs_bigr[k])
	#for k in trigr:
	#	print(k, trigr[k], probs_trigr[k])
	
	print('Sentences generated using unigrams:')
	for i in range(10):
		generate_sentence(unigr)

	print('\nSentences generated using bigrams:')
	for i in range(10):
		generate_sentence(unigr, bigr)

	print('\nSentences generated using trigrams:')
	for i in range(10):
		generate_sentence(unigr, bigr, trigr)

	print('\nSentences generated using 4-grams:')
	for i in range(10):
		generate_sentence(unigr, bigr, trigr, fourgr)

	print('\nSentences generated using 5-grams:')
	for i in range(10):
		generate_sentence(unigr, bigr, trigr, fourgr, fivegr)

	print('\nSentences generated using 6-grams:')
	for i in range(10):
		generate_sentence(unigr, bigr, trigr, fourgr, fivegr, sixgr)


