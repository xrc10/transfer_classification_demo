# transfer_classification_demo

The scripts run a demo for cross-lingual text classification(CLTC), with training documents in English and testing documents in French and Hausa. The task is to translate a naive bayes model trained on English to target language and test on target language documents. The inputs of the system are a bilingual dictionary between English and target language and English and target language documents(with topic labels). The output of the system is a naive bayes classifier that can be applied on target language documents.

The scripts runs probabilistic translation of naive bayes model:
- runCrossLingualDemo.m: the example code for running funRcvNBTransLearn and evaluating classification performance for cross-lingual text classification.
- funRcvNBTransLearn.m: the main function to train, translate and evaluate our CLTC method, the input is a dictionary matrix and language name(in this demo is French and Hausa).

All data lies in the data/`<target_language>` folder:
- simM.mat: the extended dictionary from link_predict demo
- goldSimM.mat: gold dictionary for trainning in link_predict demo

Folders NBmodels/ NBmodels_bolt/ stores temporary naive bayes models.

Folders dataSplit/ dataSplit_bolt/ stores documents in English and target language and their split for training, validation and testing.
 


