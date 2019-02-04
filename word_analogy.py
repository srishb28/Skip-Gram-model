import os
import pickle
import numpy as np
#import tensorflow as tf
#from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
model_path = './model/'
loss_model = 'nce'
#loss_model = 'cross_entropy'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))


dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
filepath='./word_analogy_dev.txt'
#filepath1='./predicted_file_batch.txt'
example_set=[]
choices_set=[]
#cosine_sim=[]
f=open("test_nce.txt","w")
#avg_set=[]


with open (filepath) as fp:
     contents = fp.readlines()
fp.close
for c in contents:
     differences_exp=[]
     
     c=c.replace("\"","")
     c=c.replace("\n","")
     examples=c.split('||')
     #print(examples)
     example_words=examples[0].split(',')
     #print(example_words)
     choice_words=examples[1].split(',')
     #print(choice_words)
     for i in example_words:
         
         pairs=i.split(':')
         #print(pairs[0])
         differences_exp.append(np.subtract(embeddings[dictionary[pairs[0]]],embeddings[dictionary[pairs[1]]]))
     avg=np.mean(differences_exp)
     choices_sim=[]
     for j in choice_words:
         pairs_choice=j.split(':')
         diff_choice=np.subtract(embeddings[dictionary[pairs_choice[0]]],embeddings[dictionary[pairs_choice[1]]])
         cosine_sim=(1-(distance.cosine(diff_choice,avg)))  
         f.write('"{}"'.format(j)+ " ")
         choices_sim.append(cosine_sim)
     #print(choices_sim)
     most_sim=(choice_words[choices_sim.index(max(choices_sim))])
     least_sim=(choice_words[choices_sim.index(min(choices_sim))])
     f.write('"{}"'.format(least_sim)+ " ")
     f.write('"{}"'.format(most_sim))
     f.write("\n")
f.close()
#     
#
#             
#         
       