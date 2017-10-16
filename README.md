# dga-isot

Project Title: Detecting Broad Length Algorithmically Generated Domains
The project aims to implement feature extraction of algorithmically generated domain names using concepts of Information theory.We measure the amount of information conveyed by the domain by analyzing character n-grams and computing the corresponding entropy. At the same, we leverage basic lexical and linguistic characteristics of the domain names which have proven effective at detecting DGAs.

Getting Started

Prereq: Python , weka 3.8

Dataset:
We used in this work a global dataset of 100,000 legitimate domains consisting of legitimate domain names from alexa.com, and a dataset of 100,000 DGA domains, giving a complete dataset of 200,000 domain names

Running:
Our feature model involves 6 features- 4 primitive features(length, vowels, consonants, digits), and two advanced features consisting of the domain n-gram entropy and conditional probability. 
Analysis of different kinds of n-grams on sample data showed that n=3 yields better results. So, we analyze and use trigrams in our feature model.  
The domain trigram entropy is computed using word segmentation. First, we extract the SLD of domain d, and for each SLD we compute the entropy of the domain based on the trigram frequency. We derive trigram frequencies from the Google n-corpus. Google n-corpus is a large database created to help machine learning initiatives across the world.
The conditional probability P(Y|X) quantifies the outcome of a random variable Y given that the value of another random variable X is known. Given a trigram tr in Trigram(L), let count(t_r,L) denote the total occurrence number of tr in Trigram(L). The conditional probability of trigram tr with respect to length l is calculated as follows:
P(t_r |l)=(count(t_r,L))/(|Trigram(L)|)	

Where |Trigram(L)| denote the size (or cardinality) of Trigram(L).

This is the basic idea behind the calculations.


Code:
feature_extraction.py can be run to extract features such as length,vowels,consonants and entropy
conditional_probability.py involves splitting the domains into trigrams and calculating the individual probability of ocurrences.



Machine learning using Weka :

-	J48 decision tree
-	Random Forests algorithm
We run the split model using threshold length l=10 on a larger dataset consisting of 200,000 domain names, with 50% legitimate domains and 50% DGA domains.  We run the experiment using 5-fold cross validation.


 Performance of Split Model on large dataset – subset of domains of length < 10
Algorithm	Detection rate (%)	FPR (%)
J48	          95.72	            4.4
Random Forest	99.24	            0.7




Performance of Split Model on large dataset – subset of domains of length ≥ 10
Algorithm	Detection rate (%)	FPR (%)
J48	          98.94	          1.5
Random Forest	98.69	          1.4



The Random forests algorithm achieves the best performance, and it performs almost similarly on shorter and longer domains. So the split model allows smooth transition, in quasi transparent way.
