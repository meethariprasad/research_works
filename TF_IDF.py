#A simple file for TFIDF. I created this because, TFIDF of sklearn was not giving expected results by reducing TFIDF score for frequent 
#words and increasing for rare words for some reason. Instead of debug/ I though build my own.
#Usage: Place this file in your execution diectory. https://github.com/meethariprasad/phd/blob/master/TF_IDF.py
# In your code add 
#from TF_IDF import *
#Input: List of text documents
#get_tf_idf_frame(documents,binary_tf=True) 
#Output: TFIDF Dataframe. Pre processing can be altered in doclist2_cleanlist

def doclist2_cleanlist(document_list):
    clean_doc_list=[]
    for doc_id,doc in enumerate(document_list):
        import nltk
        sent_text = nltk.sent_tokenize(doc) # this gives us a list of sentences
        clean_sent_list=[]
        for sentence in sent_text:
            sentence=sentence.lower()
            import re
            clean_sent=re.sub('[^A-Za-z0-9_-]+', ' ', sentence).strip()
            clean_sent_list.append(clean_sent)
        clean_doc_list.append(' '.join(clean_sent_list))
    return(clean_doc_list)

def tf_of_words_per_doc(documents):
    list_of_unique_words_in_documents=list(set(' '.join(documents).split()))
    doc_word_TF_Doc_word=[]
    for word in list_of_unique_words_in_documents:
        for doc_id,doc in enumerate(documents):
            Number_of_words_in_current_doc=len(doc.split())
            def count_if_found(str, word):
                a = str.split(" ") 
                count = 0
                for i in range(0, len(a)): 
                    if (word == a[i]): 
                       count = count + 1 
                return count 
            number_of_time_word_repeated_in_current_doc=count_if_found(doc,word)
            import math
            TF_Doc1_word=math.log(1+(number_of_time_word_repeated_in_current_doc/Number_of_words_in_current_doc))
            doc_word_TF_Doc_word.append(list([doc_id,word,TF_Doc1_word]))
    return(doc_word_TF_Doc_word)

def idf_words(documents):
    import math
    Number_of_documents=len(documents)
    idf_words=[]
    
    def number_of_documents_word_appears(word,documents):
        number_of_documents_word_appears_count=0        
        for doc in documents:
            unique_words_per_doc=(list(set(doc.split())))
            for doc_word_set in unique_words_per_doc:
                if word in doc_word_set:
                    number_of_documents_word_appears_count=number_of_documents_word_appears_count+1
        return(number_of_documents_word_appears_count)
    
    list_of_unique_words_in_documents=list(set(' '.join(documents).split()))
    
    for word in list_of_unique_words_in_documents:
        number_of_documents_word_appears_count=number_of_documents_word_appears(word,documents)
        IDF_Doc_word=math.log(Number_of_documents/(1+number_of_documents_word_appears_count))+1
        idf_words.append(list([word,Number_of_documents,number_of_documents_word_appears_count,IDF_Doc_word]))

    return(idf_words)

def get_tf_idf_frame(documents,binary_tf):
    import pandas as pd
    import numpy as np
    documents=doclist2_cleanlist(documents)
    TF_Frame=pd.DataFrame(tf_of_words_per_doc(documents),columns=["doc","word","TF"])
    IDF_Frame=pd.DataFrame(idf_words(documents),columns=["word","Number_of_documents","number_of_documents_word_appears_count","idf"])
    TF_IDF_Master_Frame=TF_Frame.merge(IDF_Frame,how="left",left_on="word",right_on="word").sort_values("doc")
    TF_IDF_Master_Frame["Term_Existence"]=np.where(TF_IDF_Master_Frame['TF']>0, 1, 0)
    if (binary_tf==True):
        TF_IDF_Master_Frame["TF_IDF"]=TF_IDF_Master_Frame["Term_Existence"]*TF_IDF_Master_Frame["idf"]
    else:
        TF_IDF_Master_Frame["TF_IDF"]=TF_IDF_Master_Frame["TF"]*TF_IDF_Master_Frame["idf"]
    return (TF_IDF_Master_Frame)
