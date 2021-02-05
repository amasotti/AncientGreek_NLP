# Skip Gram Negative Model for Homer

This project contains my PyTorch implementation of the Word2Vec (Skip Gram) trained on the Homeric texts (Ilias and Odyssey).

The original Neural Model (`modules/CBOW`) was inspired by [ n0obcoder /
Skip-Gram-Model-PyTorch ](https://github.com/n0obcoder/Skip-Gram-Model-PyTorch) and then heavily modified.

## Folder Structure


## Pipeline & Parameters

+ *Data preprocessing*: The homeric texts were merged, then cleaned and the stopwords deleted (see utilities in `utils/utils`)
    Since this is quite an expensive task, the ready-to-go files were saved (.json and .npy) and can be directly loaded when
    training the Neural Network. 
+ *Dataset* : From the cleaned data a Skip Gram dataset was extracted
        
        *example here**
+ *SkipGram Implementation* : Finally a Pytorch Module for the Skipgram was implemented and trained. The model consists of two Emdebbings layers (one for input, the other for context words)
    The activation function is the LogSigmoid.
    
    **Parameters**:
    
     + batch_size: 1024*3
     + learning_rate : 0.001, improved while training using the `ReduceOnPlateau` scheduler, with `patience=1`, `factor=0.3`
     + optimizer : Adaptive Optimizer `Adam`
     + Loss Function: 
     

### Loss curve

#### Training phase
![loss train](./losses_train.png =250x250)

#### Validation Phase
![loss train](./losses_val.png =250x250)



### Predictions

|      θεά      |    εὔχομαι    |       ἔρος    | ερχομαι       |
| :-----------: | :-----------: | :-----------: | :-----------: |
| γλαυκῶπις     | ειναι         | περιπροχυθεις |δομενευ        |
| ἀθήνη         | εξειπω        | ρυμνης        |γενεσιν        |
| Ἥρη           | νικησαντʼ     | γυναικος      |οτρυνῃσιν      |
| θύμῳ          | εγω           | καταλεξομεν   |απασης         |
| περ           | ος            | ουδε          |βουληφορε      |
