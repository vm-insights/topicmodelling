# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:24:06 2020

@author: VINAY MENON
"""

from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV

def lda_gridsearch(search_params, word_tf):
    # Init the Model
    lda = LatentDirichletAllocation()
    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)
    # Do the Grid Search
    model.fit(word_tf)    
    
    return model