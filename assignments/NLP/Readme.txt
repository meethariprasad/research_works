Author: Hari Prasad, IIITH Rollnumber: 20173074, PGSSP

Environment: Numpy ('1.13.1'), Tensorflow,NLTK latest version. (Anaconda Distribution was used)

Folder Structure:

Folder: 20173074_assignment1

	20173074_cbow
	20173074_skipgram

		Folder: negative_sampling
			20173074_cbow
			20173074_skipgram

Run by going to appropriate folder: python filename.py

What this folder implements?

This is the file which implements Tomas Mikolov Word2Vec "Efficient Estimation of Word Representations in Vector Space" https://arxiv.org/pdf/1301.3781.pdf,
which was further simplified by Xin Rong in "word2vec Parameter Learning Explained", https://arxiv.org/pdf/1411.2738.pdf.

What about these implementation?

Word2Vec algorithm is computationally expensive, considering the vocabulary size and embeddings.

In this implementation, the corpos given is trained for 1,00,000 iterations for 3 days in 8 GB Ram and Quadcore CPU. 

These learned embeddings are saved and shared in google drive.

User will be giving multiple target words in comma seperated manner.

User is also given the option to download the embeddings or wait for training by entering 1 or 0.

The code in general will clean the corpus, will build the dictionary of indexed words, will created indexed sequence.

One hot encodings will be created in the form of label and context or context and label, and will be fead in to learning algorithm.

Learning algorithm is implemented based on vectorization of equations.

The naive implementations in main folder implements everything from forward as well as backward propogation step by step.

Tensorflow implementation in negative sampling folder implements, negative sampling, sampled softmax, AdamGram optimizer, batch learning making it more sophisticated and 
computationally (feasible) to learn and predict.

Every file can be run independently.

Note: Advice for the user is to choose download the embedding for predictions using option 1 and go through the flow of program by opening program file.

References: 

:Xin Rong in "word2vec Parameter Learning Explained", https://arxiv.org/pdf/1411.2738.pdf.
:Udacity Learnings & blogs