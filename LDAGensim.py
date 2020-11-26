# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 20:07:30 2020

@author: VINAY MENON
"""
import gensim
from gensim.corpora import Dictionary
from gensim import corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
    

def lda_gensim_model(df_data_lemmatized):
    text_list = [i.split() for i in df_data_lemmatized['text'].tolist()]
    id2word = Dictionary(text_list)
    doc_term_matrix = [id2word.doc2bow(text) for text in text_list]
    corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)
    start =2
    limit = 30
    step =1
    coherence_values = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaModel(doc_term_matrix, num_topics, id2word)
        cm = CoherenceModel(model=model, 
                            corpus=doc_term_matrix, 
                            coherence='u_mass')
        coherence_values.append(cm.get_coherence())

    # Show graph
    #limit=11; start=2; step=1;
    x = range(start, limit, step)
    print(x)
    print(coherence_values)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


    
    
