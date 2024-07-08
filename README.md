# FeedbackControlledRNN
Code to simulate BCI task and adaptation in feedback-controlled RNNs. Accompanies the following preprint:

Tested with Python 3.9.6 and Python 3.10.12

For details, look at:
- [Model construction](#model-construction)


## Model construction
To create and train new models, use the `scripts/batch_create_sparse.py` file. You can specify various options for construction, including foldername `-F` and number of models `-nf`. The folder will be inside [/use_models/](/use_models)

**Example 1:** Simple velocity decoder
```
$ python scripts/batch_create_network.py -F 'relu_/' -nf 5 -te 400 -pe 250
```

**Example 2:** Simple position decoder
```
$ python scripts/batch_create_network.py -F 'relu_/' -nf 5 -te 400 -pe 250 -decode_p 1
```

**Example 3:** Simple velocity decoder with sparse weights
```
$ python scripts/batch_create_network.py -F 'relu_sparse_wt_/' -nf 5 -te 400 -pe 250 -wt 0.3
$ python scripts/batch_create_network.py -F 'relu_sparse_neu_/' -nf 5 -te 400 -pe 250 -neu 0.3
```

**Example 3:** Simple velocity decoder with sparse weights
```
$ python scripts/batch_create_network.py -F 'relu_sparse_wt_/' -nf 5 -te 400 -pe 250 -wt 0.3
$ python scripts/batch_create_network.py -F 'relu_sparse_neu_/' -nf 5 -te 400 -pe 250 -neu 0.3
```


