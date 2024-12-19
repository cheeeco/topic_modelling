import artm

batch_vectorizer = artm.BatchVectorizer(data_path='semantic_segmentation', data_format='bow_uci',
                                        collection_name='bow', target_folder='bow_batches')

dictionary = batch_vectorizer.dictionary

# EXPERIMENT 1. Primitive artm.LDA model
print("EXPERIMENT 1. Primitive artm.LDA model")

model_lda = artm.LDA(num_topics=5, cache_theta=True)
model_lda.initialize(dictionary=dictionary)

model_lda.num_document_passes = 1

model_lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)

print("Extracted topics:")
top_tokens = model_lda.get_top_tokens(num_tokens=10)
for i, token_list in enumerate(top_tokens):
     print('Topic #{0}: {1}'.format(i, token_list))


# EXPERIMENT 2. artm.ARTM model with default parameters
print("EXPERIMENT 2. artm.ARTM model with default parameters")

topic_names = ['Topic #{}'.format(i) for i in range(5)]
model_artm = artm.ARTM(topic_names=topic_names, cache_theta=True,)
model_artm.initialize(dictionary=dictionary)

model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore'))
model_artm.num_document_passes = 1

model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=30)

print("Extracted topics:")
for topic_name in model_artm.topic_names:
    print(topic_name + ': ', end=' ')
    print(model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name])


# EXPERIMENT 3. artm.ARTM model trained on TF-IDF dataset
print("EXPERIMENT 3. artm.ARTM model with default parameters")

batch_vectorizer = artm.BatchVectorizer(data_path='semantic_segmentation', data_format='bow_uci',
                                        collection_name='tfidf', target_folder='tfidf_batches')

dictionary = batch_vectorizer.dictionary

topic_names = ['Topic #{}'.format(i) for i in range(5)]
model_artm = artm.ARTM(topic_names=topic_names, cache_theta=True,)
model_artm.initialize(dictionary=dictionary)

model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore'))
model_artm.num_document_passes = 1

model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=30)

print("Extracted topics:")
for topic_name in model_artm.topic_names:
    print(topic_name + ': ', end=' ')
    print(model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name])