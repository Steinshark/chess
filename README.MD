# Learning Chess Through Self-Play
This project is my attempt to create an AlphaZero-like chess engine. Some focus areas of the project include:
- [ ] Understanding the novel Reinforcement Learning design of the original AlphaZero algorithm.
- [ ] Optimizing hyperparameters and settings to achieve noticeable performance on consumer hardware
- [ ] Optimizing code for high performance
- [ ] Understanding the challenges of ML dealing with high GPU <-> CPU dependencies

## The Algorithm:
The basic approach of this algorithm is to replace a traditional tree search of a given chess position with a guided search using a neural network.
For a given position, the network outputs a policy and a value estimation. Using these values, the tree of possible next-moves can be explored by an
exploration vs exploitation style heuristic, namely PUCT, to create a MCTS effect for a given number of explorations down the tree per position (~800 used).
The resulting move counts are used to train the network's policy estimator un-supervised. The final game outcome is used to train the value estimator.
This is all [detailed in the DeepMind paper](https://arxiv.org/abs/1712.01815).


## The Architecture:
The chess position is encoded in a 19x8x8 tensor where the first 14 channels correspond to the existence of a particular
piece on each of the 64 squares. The next 5 channels encode castling rights, and the final channel encodes turn.

The model used in my version of the algorithm is curated for chess and uses 3 types of Conv2d layers per layer:
- [ ] Horizontal 1x15 filters (responding to chess board ranks)
- [ ] Vertical 15x1 filters (responding to chess board files)
- [ ] Traditional 7x7 Conv2d filters (aggregating everything else)

The input is passed into each filter and the outputs are concatenated to form the next input.
There are 3 such layers in the current architecture. The final (4th) layer of the model applies
32 3x3 Conv2d filters to the final output and flattens the output for the heads.

The network outputs via 2 heads. The policy head:
- [ ] Linear
- [ ] Softmax (output size 1968, correspond to all legal uci moves)

And the value head:
- [ ] Linear
- [ ] Dropout
- [ ] Linear
- [ ] Tanh




### Conv2d Block (Vertical oriented):
```python
self.vconv1         = torch.nn.Sequential(torch.nn.Conv2d(in_ch,v_conv_n,kernel_size=(7+7+1,1),stride=1,padding=(7,0),bias=False),
                                          torch.nn.BatchNorm2d(v_conv_n),
                                          conv_act())
```  
## Results
This project is a work in progress! It has been 4 weeks since I started. Interested in contributing? Reach out, I'd love to collaborate!
