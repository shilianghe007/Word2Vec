# Word2Vec
Implementing the Word2Vec model in C++ (for learning word embeddings)

This project is an implementation of Word2Vec. Following peghoty’s blog (https://www.cnblogs.com/peghoty/p/3857839.html), I studied both the mathematical principles behind Word2Vec and its code structure, and then reproduced the model using C++.

The project implements the CBOW and Skip-gram models with Hierarchical Softmax, as well as the CBOW model with Negative Sampling.

Corpus:
A lightweight Wikipedia-based corpus with over 10 million tokens was used. Since parallel training has not been implemented yet, the current single-threaded training takes roughly an entire day to complete.

The corpus has been uploaded to Baidu Netdisk:
Link: https://pan.baidu.com/s/1RbD1umuJX5mggU13Gz4nRw
Extraction code: pjmk
