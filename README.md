# GreekParser

Miscellaneous folder, my personal NLP Playground with Ancient Greek Texts.

**DISCLAIMER** : Many of these files are Work in Progress and many corrections are surely needed. Feel free to explore, I would
be thankful for suggestions!

## Structure

+ `Auxiliary` (mostly utils for loading and cleaning greek texts - preprocessing)
+ `tests`
    + word2vec_homer : a jupyter notebook with my Word2Vec model for the homeric texts
    + fastText : my fastText model for the homeric texts
+ `treebank` : xml data from Perseus
+ `data` : raw data, json vocabs, morphological parsings extracted from the Ilias and Odyssey
+ `wordEmbeddings` : Tests with CBOW and SkipGram models. 
    + `gensim_models` : word Embedding models with the Gensim library
    + `skipGram_pytorch` : my implementation of the skipgram model, trained on the homeric texts. 