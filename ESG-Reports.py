# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:11:00 2020

@author: VINAY MENON
"""
# %% Libraries
import pandas as pd
import io
import requests
import PyPDF2
import spacy
import string
import re
import gensim
from sklearn.feature_extraction import text
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import seaborn as sns
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
from pyLDA_visual import topic_visual
from Grid_Search_LDA import lda_gridsearch
from LDAGensim import lda_gensim_model
from sklearn.preprocessing import MinMaxScaler

# %% Hardcoded values
esg_urls_rows = [
  ['barclays', 'https://home.barclays/content/dam/home-barclays/documents/citizenship/ESG/Barclays-PLC-ESG-Report-2019.pdf'],
  ['jp morgan chase', 'https://impact.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/documents/jpmc-cr-esg-report-2019.pdf'],
  ['morgan stanley', 'https://www.morganstanley.com/pub/content/dam/msdotcom/sustainability/Morgan-Stanley_2019-Sustainability-Report_Final.pdf'],
  ['goldman sachs', 'https://www.goldmansachs.com/what-we-do/sustainable-finance/documents/reports/2019-sustainability-report.pdf'],
  ['hsbc', 'https://www.hsbc.com/-/files/hsbc/our-approach/measuring-our-impact/pdfs/190408-esg-update-april-2019-eng.pdf'],
  ['citi', 'https://www.citigroup.com/citi/about/esg/download/2019/Global-ESG-Report-2019.pdf'],
  ['td bank', 'https://www.td.com/document/PDF/corporateresponsibility/2018-ESG-Report.pdf'],
  ['bank of america', 'https://about.bankofamerica.com/assets/pdf/Bank-of-America-2017-ESG-Performance-Data-Summary.pdf'],
  ['rbc', 'https://www.rbc.com/community-social-impact/_assets-custom/pdf/2019-ESG-Report.PDF'],
  ['macquarie', 'https://www.macquarie.com/assets/macq/investor/reports/2020/sections/Macquarie-Group-FY20-ESG.pdf'],
  ['lloyds', 'https://www.lloydsbankinggroup.com/globalassets/documents/investors/2020/2020feb20_lbg_esg_approach.pdf'],
  ['santander', 'https://www.santander.co.uk/assets/s3fs-public/documents/2019_santander_esg_supplement.pdf'],
  ['bluebay', 'https://www.bluebay.com/globalassets/documents/bluebay-annual-esg-investment-report-2018.pdf'],
  ['lasalle', 'https://www.lasalle.com/documents/ESG_Policy_2019.pdf'],
  ['riverstone', 'https://www.riverstonellc.com/media/1196/riverstone_esg_report.pdf'],
  ['aberdeen standard', 'https://www.standardlifeinvestments.com/RI_Report.pdf'],
  ['apollo', 'https://www.apollo.com/~/media/Files/A/Apollo-V2/documents/apollo-2018-esg-summary-annual-report.pdf'],
  ['bmogan', 'https://www.bmogam.com/gb-en/intermediary/wp-content/uploads/2019/02/cm16148-esg-profile-and-impact-report-2018_v33_digital.pdf'],
  ['vanguard', 'https://personal.vanguard.com/pdf/ISGESG.pdf'],
  ['ruffer', 'https://www.ruffer.co.uk/-/media/Ruffer-Website/Files/Downloads/ESG/2018_Ruffer_report_on_ESG.pdf'],
  ['northern trust', 'https://cdn.northerntrust.com/pws/nt/documents/fact-sheets/mutual-funds/institutional/annual-stewardship-report.pdf'],
  ['hermes investments', 'https://www.hermes-investment.com/ukw/wp-content/uploads/sites/80/2017/09/Hermes-Global-Equities-ESG-Dashboard-Overview_NB.pdf'],
  ['abri capital', 'http://www.abris-capital.com/sites/default/files/Abris%20ESG%20Report%202018.pdf'],
  ['schroders', 'https://www.schroders.com/en/sysglobalassets/digital/insights/2019/pdfs/sustainability/sustainable-investment-report/sustainable-investment-report-q2-2019.pdf'],
  ['lazard', 'https://www.lazardassetmanagement.com/docs/-m0-/54142/LazardESGIntegrationReport_en.pdf'],
  ['credit suisse', 'https://www.credit-suisse.com/pwp/am/downloads/marketing/br_esg_capabilities_uk_csam_en.pdf'],
  ['coller capital', 'https://www.collercapital.com/sites/default/files/Coller%20Capital%20ESG%20Report%202019-Digital%20copy.pdf'],
  ['cinven', 'https://www.cinven.com/media/2086/81-cinven-esg-policy.pdf'],
  ['warburg pircus', 'https://www.warburgpincus.com/content/uploads/2019/07/Warburg-Pincus-ESG-Brochure.pdf'],
  ['exponent', 'https://www.exponentpe.com/sites/default/files/2020-01/Exponent%20ESG%20Report%202018.pdf'],
  ['silverfleet capital', 'https://www.silverfleetcapital.com/media-centre/silverfleet-esg-report-2020.pdf'],
  ['kkr', 'https://www.kkr.com/_files/pdf/KKR_2018_ESG_Impact_and_Citizenship_Report.pdf'],
  ['cerberus', 'https://www.cerberus.com/media/2019/07/Cerberus-2018-ESG-Report_FINAL_WEB.pdf'],
  ['standard chartered', 'https://av.sc.com/corp-en/others/2018-sustainability-summary2.pdf'],
]

# %% Extracting PDF data into dataframe

complete_list = []
# pdf extract process
def extract_content(url):
  """
  A simple user define function that, given a url, download PDF text content
  Parse PDF and return plain text version
  """
  try:
    # retrieve PDF binary stream
    response = requests.get(url)
    open_pdf_file = io.BytesIO(response.content)
    pdf = PyPDF2.PdfFileReader(open_pdf_file)  
    # access pdf content
    text = [pdf.getPage(i).extractText() for i in range(0, pdf.getNumPages())]
    # return concatenated content
    return "\n".join(text)
  except:
    return ""

for i in range(len(esg_urls_rows)):
    text_data = extract_content(esg_urls_rows[i][1])
    complete_list.append([esg_urls_rows[i][0], esg_urls_rows[i][1], text_data])
    
# create a Pandas dataframe of ESG report URLs
esg_urls_pd = pd.DataFrame(complete_list, columns=['company', 'url', 'text'])

# %% Load spacy model

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm", disable=['ner'])

# %%NLP : Extracting proper statments

def remove_non_ascii(text):
  printable = set(string.printable)
  return ''.join(filter(lambda x: x in printable, text))

def not_header(line):
  # as we're consolidating broken lines into paragraphs, we want to make sure not to include headers
  return not line.isupper()


def extract_statements(nlp, company, text):
  """
  Extracting ESG statements from raw text by removing junk, URLs, etc.
  We group consecutive lines into paragraphs and use spacy to parse sentences.
  """
  lines = []
  sentences = []
  # remove non ASCII characters
  text = remove_non_ascii(text)

  prev = ""
  for line in text.split('\n'):
    # aggregate consecutive lines where text may be broken down
    # only if next line starts with a space or previous does not end with dot.
    if(line.startswith(' ') or not prev.endswith('.')):
        prev = prev + ' ' + line
    else:
        # new paragraph
        lines.append(prev)
        prev = line
        
  # don't forget left-over paragraph
  lines.append(prev)
  
  # clean paragraphs from extra space, unwanted characters, urls, etc.
  # best effort clean up, consider a more versatile cleaner

  for line in lines:
    
      # removing header number
      line = re.sub(r'^\s?\d+(.*)$', r'\1', line)
      # removing trailing spaces
      line = line.strip()
      # words may be split between lines, ensure we link them back together
      line = re.sub('\s?-\s?', '-', line)
      # remove space prior to punctuation
      line = re.sub(r'\s?([,:;\.])', r'\1', line)
      # ESG contains a lot of figures that are not relevant to grammatical structure
      line = re.sub(r'\d{5,}', r' ', line)
      # remove mentions of URLs
      line = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', r' ', line)
      # remove multiple spaces
      line = re.sub('\s+', ' ', line)
      # split paragraphs into well defined sentences using spacy
      for part in list(nlp(line).sents):
        sentences.append([company, str(part).strip()]) 

  return sentences

statement_list = []
for i in range(len(complete_list)):
    company = complete_list[i][0]
    statements = extract_statements(nlp, company, complete_list[i][2])
    statement_list.extend(statements)
    
# %% NLP : Lemmatization - singular , present form

def tokenize(sentence):
  gen = gensim.utils.simple_preprocess(sentence, deacc=True)
  return ' '.join(gen)

def lemmatize(nlp, text):
  
  # parse sentence using spacy
  doc = nlp(text) 
  
  # convert words into their simplest form (singular, present form, etc.)
  lemma = []
  for token in doc:
      if (token.lemma_ not in ['-PRON-']):
          lemma.append(token.lemma_)
          
  return tokenize(' '.join(lemma))

stat_lem_list = []
for i in range(len(statement_list)):
    company = statement_list[i][0]
    stat_lem = lemmatize(nlp, statement_list[i][1])
    stat_lem_list.append([company, stat_lem])

# create dataframe
esg_lem_data = pd.DataFrame(stat_lem_list, columns=['company', 'text'])

# %% Stop words

# context specific keywords not to include in topic modelling
fsi_stop_words = [
  'plc', 'group', 'target',
  'track', 'capital', 'holding',
  'report', 'annualreport',
  'esg', 'bank', 'report',
  'annualreport', 'long', 'make'
]

# add company names as stop words
for fsi in [row[0] for row in esg_urls_rows]:
    for t in fsi.split(' '):
        fsi_stop_words.append(t)

# our list contains all english stop words + companies names + specific keywords
stop_words = text.ENGLISH_STOP_WORDS.union(fsi_stop_words)

# %% word cloud
# aggregate all 7200 records into one large string to run wordcloud on term frequency
# we could leverage spark framework for TF analysis and call wordcloud.generate_from_frequencies instead
large_string = ' '.join(esg_lem_data.text)

# use 3rd party lib to compute term freq., apply stop words
word_cloud = WordCloud(
    background_color="white",
    max_words=5000, 
    width=900, 
    height=700, 
    stopwords=stop_words, 
    contour_width=3, 
    contour_color='steelblue'
)

# display our wordcloud across all records
plt.figure(figsize=(10,10))
word_cloud.generate(large_string)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# %% Bigram - analysis
# Run bi-gram TF-IDF frequencies
bigram_tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(2,2), min_df=10, use_idf=True)
bigram_tf_idf = bigram_tf_idf_vectorizer.fit_transform(esg_lem_data.text)

# Extract bi-grams names
words = bigram_tf_idf_vectorizer.get_feature_names()

# extract our top 10 ngrams
total_counts = np.zeros(len(words))
for t in bigram_tf_idf:
    total_counts += t.toarray()[0]

count_dict = (zip(words, total_counts))
count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
words = [w[0] for w in count_dict]
counts = [w[1] for w in count_dict]
x_pos = np.arange(len(words)) 

# Plot top 10 ngrams
plt.figure(figsize=(15, 5))
plt.subplot(title='10 most common bi-grams')
sns.barplot(x_pos, counts, palette='Blues_r')
plt.xticks(x_pos, words, rotation=90) 
plt.xlabel('bi-grams')
plt.ylabel('tfidf')
plt.show()

# %% Modeling : NMF (Non Negative Matrix factorization)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

word_tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,1))
word_tf_idf = word_tf_idf_vectorizer.fit_transform(esg_lem_data.text)
n_top_words = 20

# Fit the NMF model Frobenius norm
print("Fitting the NMF model (Frobenius norm) with tf-idf feature")
nmf = NMF(n_components=15, random_state=42,
          alpha=.3, l1_ratio=.5).fit(word_tf_idf)

print("\nTopics in NMF model (Frobenius norm):")
tfidf_feature_names = word_tf_idf_vectorizer.get_feature_names()
print_top_words(nmf, tfidf_feature_names, n_top_words)

# %% Modeling: LDA

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

word_tf_vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1,1))
word_tf = word_tf_vectorizer.fit_transform(esg_lem_data.text)

# Build LDA Model
lda_model = LatentDirichletAllocation(n_components=9, 
                                      learning_decay=0.3,
                                      #max_iter=10,               # Max learning iterations
                                      #learning_method='online',   
                                      random_state=42,          # Random state
                                      #batch_size=128,            # n docs in each learning iter
                                      #evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                      #n_jobs = -1,               # Use all available CPUs
                                     )
lda_output = lda_model.fit(word_tf)

print(lda_model)
# Log Likelyhood: Higher the better
print("Log Likelihood: ", lda_model.score(word_tf))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(word_tf))

# See model parameters
print(lda_model.get_params())

print_top_words(lda_model, word_tf_vectorizer.get_feature_names(), 20)

# %% Grid Search
# Define Search Param
search_params = {'n_components': [3, 5, 7, 9, 11],
                 'learning_decay': [.3, .5, .7, .9],
                 'random_state': [20, 40, 60, 80]}

model = lda_gridsearch(search_params, word_tf)
best_lda_model = model.best_estimator_
# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(word_tf))

# Best Model's Params:  {'learning_decay': 0.3, 'n_components': 3, 'random_state': 20}
# Best Log Likelihood Score:  -235851.2030676004
# Model Perplexity:  2119.0453620682456
# %% pyLDAvis
    
topic_visual(lda_model,
             word_tf,
             word_tf_vectorizer
             )


# %% Gensim LDA for coherence score
lda_gensim_model(esg_lem_data) #(stat_lem_list)
# No. of Topics = 9
# Human intution of topic names:
#TOPIC 1(G) : Ethical investment
#TOPIC 2(E): Sustainable finance
#TOPIC 3(S): Value employee
#TOPIC 4(G): Code of Conduct
#TOPIC 5(E): Climate change
#TOPIC 6(E): renewable energy
#TOPIC 7(G): Customer centric
#Topic 8(G): Strong governance
#Topic 9(S): Support community
topic_names = [
  'ethical investment',
  'Sustainable finance', 
  'Value employee',
  'Code of Conduct',
  'Climate change',
  'renewable energy',
  'Customer centric',
  'Strong governance',
  'Support community'
]

#%% Topic Distribution for each of the statements

transformed = lda_model.transform(word_tf)
# find principal topic from distribution...
a = [topic_names[np.argmax(distribution)] for distribution in transformed]
# ... with associated probability
b = [np.max(distribution) for distribution in transformed]

esg_prob = pd.DataFrame(zip(a,b,transformed), columns=['topic', 'probability', 'probabilities'])
esg_lem_data_prob = pd.concat([esg_lem_data, esg_prob], axis=1)

#%% Compare Companies over ESG Initiatives

# create a simple pivot table of number of occurence of each topic across organisations
esg_focus = pd.crosstab(esg_lem_data_prob.company, esg_lem_data_prob.topic)

# scale topic frequency between 0 and 1
scaler = MinMaxScaler(feature_range = (0, 1))

# normalize pivot table
esg_focus_norm = pd.DataFrame(scaler.fit_transform(esg_focus), columns=esg_focus.columns)
esg_focus_norm.index = esg_focus.index

# plot heatmap, showing main area of focus for each FSI across topics we learned
sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(esg_focus_norm, annot=False, linewidths=.5, cmap='Blues')
plt.show()





