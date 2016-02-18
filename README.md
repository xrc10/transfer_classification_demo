# transfer_classification_demo

The scripts run a demo for cross-lingual text classification(CLTC), with training documents in English and testing documents in French. The task is to translate a naive bayes model trained on English to French and test on French documents. The inputs of the system are a bilingual dictionary between English and French and English and French documents(with topic labels). The output of the system is a naive bayes classifier that can be applied on French documents.

The scripts runs probabilistic translation of naive bayes model:
- runCrossLingualDemo.m: the example code for running funRcvNBTransLearn and evaluating classification performance for cross-lingual text classification.
- funRcvNBTransLearn.m: the main function to train, translate and evaluate our CLTC method, the input is a dictionary matrix and language name(in this demo is French).

All data lies in the data/ folder:
- simM.mat: the extended dictionary from link_predict demo
- goldSimM.mat: gold dictionary for trainning in link_predict demo

Folder NBmodels/ stores temporary naive bayes models.

Folder dataSplit/ stores documents in English and French and their split for training, validation and testing.
 


