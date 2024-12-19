import artm

batch_vectorizer = artm.BatchVectorizer(data_path='object_detection', data_format='bow_uci',
                                        collection_name='tfidf', target_folder='bow_batches')

dictionary = batch_vectorizer.dictionary
topic_names = ['topic_{}'.format(i) for i in range(5)]

model_lda = artm.LDA(num_topics=5, cache_theta=True)
model_lda.initialize(dictionary=dictionary)

model_artm = artm.ARTM(topic_names=topic_names, cache_theta=True,)
                    #    scores=[artm.PerplexityScore(name='PerplexityScore',
                    #                                 dictionary=dictionary)],)
                    #    regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta',
                    #                                                    tau=-0.15)])
model_artm.initialize(dictionary=dictionary)

model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore'))

model_lda.num_document_passes = 1
model_artm.num_document_passes = 1

model_lda.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=30)
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=30)

print("model_lda")
top_tokens = model_lda.get_top_tokens(num_tokens=10)
for i, token_list in enumerate(top_tokens):
     print('Topic #{0}: {1}'.format(i, token_list))

print("model_artm")
for topic_name in model_artm.topic_names:
    print(topic_name + ': ')
    print(model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name])
