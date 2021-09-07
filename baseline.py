import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



def find_k(data, max_k):
	iters = range(2, max_k+1, 2)
	sse = []
	for k in iters:
		sse.append(MiniBatchKMeans(n_clusters = k, init_size = 1024, batch_size = 2048, random_state = 20).fit(data).inertia_)
	f,ax = plt.subplots(1,1)
	ax.plot(iters, sse, marker='o')
	ax.set_xlabel('Cluster Centers')
	ax.set_xticks(iters)
	ax.set_xticklabels(iters)
	ax.set_ylabel('SSE')
	ax.set_title('SSE by Cluster Center Plot')
	#plt.show()


def get_top_key_words ( data, clusters, labels, n_terms):
	df = pd.DataFrame(data.todense()).groupby(clusters).mean()

	key_words = []
	for i, r in df.iterrows():
		key_words.append(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
	return key_words



def main():
	# arr = os.listdir("/Users/moweiting/Desktop/annotation_input")
	# long_review_text = []
	# for txt in arr:
	# 	f = open("/Users/moweiting/Desktop/annotation_input/"+txt, "r")
	# 	for paragraph in f:
	# 		long_review_text.append(paragraph.lower())
	# paragraph_new = []
	# st = stopwords.words('english')
	# for sentence in long_review_text:
	#     word_list = sentence.split()
	#     #print(word_list)
	#     texts_tmp = []
	#     for word in word_list:
	#         if word not in stopwords.words('english'):
	#             texts_tmp.append(word)
	#             #print(word)
	#     texts_str = ' '.join(texts_tmp)
	#     #print(texts_str)
	#     paragraph_new.append(texts_str)
	#     with open("longreviews_sentences_nostopwords.txt", 'a', encoding='utf8') as wf:
	#         wf.write(texts_str + '\n')


	
	nonstop_text = []
	f = open("longreviews_sentences_nostopwords.txt", "r")
	for paragraph in f:
		nonstop_text.append(paragraph)



	vectorizer = TfidfVectorizer()
	vectorizer.fit(nonstop_text)
	text = vectorizer.transform(nonstop_text)

	find_k(text, 40)

	clusters = MiniBatchKMeans(n_clusters = 10, init_size = 1024, batch_size = 2048, random_state = 20).fit_predict(text)

	indices = []
	for i in range (0,10):
		indices.append ([j for j, x in enumerate(clusters) if x == i])
	
	file = open("clustered_text_10.txt","w")

	key_words = get_top_key_words(text, clusters, vectorizer.get_feature_names(),10)



	file.write("cluster: \n")
	file.write(key_words[9])
	file.write("\n")
	file.write("\n")
	for j in range (0,len(indices[9])):
		file.write(nonstop_text[indices[9][j]])
		file.write('\n')
	file.write('\n')

if __name__ == "__main__":
    main()