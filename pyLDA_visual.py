# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 17:53:08 2020

@author: VINAY MENON
"""
import pyLDAvis
from pyLDAvis import sklearn

def topic_visual(best_lda_model, data_vectorized, vectorizer):
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.sklearn.prepare(best_lda_model,
                             data_vectorized,
                             vectorizer,
                             mds='tsne')
    pyLDAvis.show(panel)
    