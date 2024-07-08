# FeedbackControlledRNN
Code to simulate BCI task and adaptation in feedback-controlled RNNs. Accompanies the following preprint:

Tested with Python 3.9.6 and Python 3.10.12

For details, look at:
- [Model construction](#model-construction)
- [Generating perturbed decoders](#generate-and-filter-perturbed-maps)


## Model construction
To create and train new models, use the `scripts/batch_create_sparse.py` file. You can specify various options for construction, including foldername `-F` and number of models `-nf`. The folder will be inside [/use_models/](/use_models)

**Example 1:** Simple velocity decoder
```
$ python scripts/batch_create_network.py -F 'relu_/' -nf 5 -te 400 -pe 250
```

**Example 2:** Simple position decoder
```
$ python scripts/batch_create_network.py -F 'relu_pos_/' -nf 5 -te 400 -pe 250 -decode_p 1
```

**Example 3:** Velocity decoder with 2-layer feedback controller module
```
$ python batch_create_network.py -F 'percp_expansion_/' -nf 2 -te 450 -pe 250 -lr 0.0002 -mtype 'layer2' -nout 4 -nhid 100 -nmf 10
```


## Generate and filter perturbed maps
Decide which models to use (follow [model construction](#model-construction) first), specify the dimensionality of the intrinsic manifold, and optionally specify conditions for filtering perturbed decoders:

Results are in [/wmp/](/wmp/), [/omp/](/omp/) or [/rmp/](/rmp/). Each model file will have its own folder with filtered WMPs saved in `WMP_maps.npy` and the combined results will be in `xmp_tested_movepc_PCK.npy` where `xmp` is wmp, omp or rmp, and `PCK` specifies the intrinsic manifold dimensionality K.

**Example 1:** Generate Within-Manifold decoders

