import tensorflow as tf
import numpy as np

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A = 
    


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
  
    #print(u_o)
    A=tf.multiply(inputs,true_w)
    #u_w=tf.reduce_sum(true_w,1)
    B=tf.log(tf.reduce_sum(tf.exp(tf.multiply(inputs,true_w)),1)+0.00000001)
    B=tf.reshape(B,[inputs.shape[0],1])
    print(B.shape)
    print(A.shape)
    #print(tf.shape(B))
    print((tf.subtract(B,A)).shape)
    return tf.subtract(B, A)

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

#    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
#    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
#    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
#    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
#    samples: Word_ids for negative samples. Dimension is [num_sampled].
#    unigram_prob: Unigram probability. Dimesion is [Vocabulary].
#
#    Implement Noise Contrastive Estimation Loss Here
#
#    ==========================================================================
#    """
    
    uo = tf.nn.embedding_lookup(weights,labels)
    uo= tf.reshape(uo, [-1, weights.shape[1]])
    
    print(uo.shape)
    b=tf.nn.embedding_lookup(biases,labels)
    #b= tf.reshape(b, [-1])
    print("bias")
    print(b.shape)
    
    s=tf.reduce_sum(tf.multiply(inputs,uo),1)
    s=tf.reshape(s,[s.shape[0],1])
    s=tf.add(s,b)
    print("s")
    print(s.shape)
    #s=tf.transpose(s)
    
    probab=(tf.nn.embedding_lookup([unigram_prob],labels))
    #probab= tf.reshape(uo, [-1, weights.shape[1]])
    k=len(sample)
    probab=tf.log(tf.scalar_mul(k,probab)+0.00000001)
    print("probab")
    print(probab.shape)
    pr1=tf.subtract(s,probab)
    pr1=tf.sigmoid(pr1)
    pr1=tf.log(pr1+0.00000001)
    print("pr1")
    print(pr1.shape)
    
    wx=tf.nn.embedding_lookup(weights,sample)
    wx=tf.reshape(wx,[sample.shape[0],-1])
    print("wx")
    print(wx.shape)
    b1=tf.nn.embedding_lookup(biases,sample)
    print("b1")
    b1=tf.reshape(b1,[b1.shape[0],1])
    print(b1.shape)
    
    print(sample.shape)
    #inputnoise=tf.nn.embedding_lookup(inputs,labels)
   
    
    #transposewx=tf.transpose(wx)
    #print("transposewx")
    #print(transposewx.shape)
    s1=tf.matmul(inputs,tf.transpose(wx))
    #print("transposewx")
    #print(transposewx.shape)
    
    s1=tf.add(s1,tf.transpose(b1))
    print("s1")
    print(s1.shape)
    #s1=tf.nn.sigmoid(s1)
    probab2=tf.nn.embedding_lookup([unigram_prob],sample)
    probab2=tf.reshape(probab2,[sample.shape[0],1])
    print("probabshape")
    print(probab2.shape)
    probab2=tf.log((len(sample)*probab2)+0.00000001)
    pr2=tf.subtract(s1,tf.transpose(probab2))
    pr2=tf.nn.sigmoid(pr2)
    pr2=tf.subtract(tf.ones([1,len(sample)]),pr2)
   
    pr2=tf.reduce_sum(tf.log(pr2+0.00000001),1)
    
    print("pr2")
    print(pr2.shape)
    final_prob=tf.negative(tf.add(pr1,pr2))
    #final_prob=tf.reshape(final_prob,[128,1])
    print("final_prob")
    print(final_prob.shape)
    return final_prob
    
