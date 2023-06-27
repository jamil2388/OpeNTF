# ``OpeNTF``: An Open-Source Neural Team Formation Benchmark Library 
Effective team formation is crucial for project success, influenced by skill coverage and geographic proximity. This paper proposes a novel approach for spatial team formation that leverages graph neural network (gnn) embeddings to transfer knowledge from a heterogenous collaborative network among experts, with metapaths and lack thereof, followed by a neural-based multi-label classifier. Our method aims to optimize team composition by considering the interplay between skill compatibility and geographic cohesiveness. Specifically, we construct a heterogeneous graph representation whose nodes are experts, skills, and locations to capture the complex relationships between experts' skills and their geographic locations. 

## :movie_camera: [`Demo`]
We propose a multi-objective optimization to guide the team formation process. We employ gnn to learn vector representations of experts, encoding both their skill profiles and geographic information using metapaths. The objective is to maximize skill coverage while minimizing geographic dispersion, balancing effective collaboration and efficient communication. We utilize classification and information retrieval metrics to evaluate the accuracy of the recommended teams of experts concerning the required skills and geographical distribution. Experimental evaluations of our proposed method on a real-world dataset of patents and computer science publications against baseline methods demonstrated the effectiveness of our approach in forming diverse and geographically cohesive teams. The findings of this study contribute to the field of team formation by highlighting the benefits of incorporating gnn embeddings considering skill and location in tandem for the task of team formation.

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Features](#3-features)
4. [Results](#4-results)
5. [Acknowledgement](#5-acknowledgement)
6. [License](#6-license)
7. [Citation](#7-citation)
8. [Awards](#8-awards)

## 1. [Setup](https://colab.research.google.com/drive/1hM2JFuLlOkkQQFcxRHAoEPyi83a_lxV2?usp=sharing)
You need to have ``3.8 <= Python <= 3.9 `` and install the following main packages, among others listed in [``requirements.txt``](requirements.txt):
```
torch>=1.9.0
pytrec-eval-terrier==0.5.2
gensim==3.8.3
```
By ``pip``, clone the codebase and install required packages:
```sh
git clone -b geo https://github.com/Fani-Lab/opentf
cd opentf
pip install -r requirements.txt
```
By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone -b geo https://github.com/Fani-Lab/opentf
cd opentf
conda env create -f environment.yml
conda activate opentf
```

For installation of specific version of a python package due to, e.g., ``CUDA`` versions compatibility, one can edit [``requirements.txt``](requirements.txt) or [``environment.yml``](environment.yml) like as follows:

```
# CUDA 10.1
torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Stellargraph library is inconsistent when installed via pip and throws errors. Hence to make installation of libraries smoother we have noted down the steps to install Stellargraph library separately. 
```
# Stellargraph
import os
!git clone https://github.com/stellargraph/stellargraph.git
os.chdir('../../stellargraph')
!pip install .
```
## 2. Quickstart [![Open In Colab](quickstart.ipynb)]

```sh
cd src
python main.py -data ../data/raw/uspt/toy.patent.tsv -domain uspt -model bnn_emb_gnn -filter 0
```

The above run, loads and preprocesses a tiny-size toy example dataset [``toy.patent.tsv``](data/raw/dblp/toy.patent.tsv) from [``uspt``](https://patentsview.org/download/data-download-tables) followed by 5-fold train-evaluation on a training split and final test on the test set for ``feedforward`` and ``Bayesian`` neural models using default hyperparameters from [``./src/param.py``](./src/param.py).

## 3. Features
**Datasets**

Raw dataset, e.g., scholarly papers from AMiner's citation network dataset of US patents from [``uspt``](https://patentsview.org/download/data-download-tables), publication data from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z)(experimentation in progress) were assumed to be populated in [``data/raw``](data/raw). For the sake of integration test, tiny-size toy example datasets [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), and [``toy.patent.tsv``](data/preprocessed/uspt/toy.patent.tsv) have been already provided.

Raw data will be preprocessed into two main ``sparse`` matrices each row of which represents: 

>i) ``vecs['member']``: occurrence (boolean) vector representation for members of a team, e.g., authors of a paper or crew members of a movie,
> 
>ii) ``vecs['skill']``: occurrence (boolean) vector representation for required skills for a team, e.g., keywords of a paper or genre of a movie.

Also, indexes will be created to map the vector's indexes to members' names and skills' names, i.e., ``i2c``, ``c2i``, ``i2s``, ``s2i``.

The sparse matrices and the indices will be persisted in [``data/preprocessed/{uspt, dblp}/{name of dataset}``](data/preprocessed/) as pickles ``teamsvecs.pkl`` and ``indexes.pkl``. For example, the preprocessed data for our dblp toy example are [``data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl``](data/preprocessed/dblp/toy.dblp.v12.json/teams.pkl) and [``data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl``](data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl).


Please note that the preprocessing step will be executed once. Subsequent runs load the persisted pickle files. In order to regenerate them, one should simply delete them. 

**[Train][Validation][Test] Split**

We randomly take ``85%`` of the dataset for the train-validation set and ``15%`` as the test set, i.e., the model never sees these instances during training or model tuning. You can change ``train_test_split`` parameter in [``./src/param.py``](./src/param.py).

We use _n_-fold cross-validation, that is, we train a model _n_ times on _(n-1)_ folds and utilize the remaining fold as the validation set to adjust the learning rate during training. The number of folds is set by ``nfolds`` in [``./src/param.py``](./src/param.py). In total, we have _n_ models each of which will be evaluated on the test set.

At each run, we store ids of instances in train-validation folds and test set in [``./data/preprocessed/{uspt,dblp}/{name of dataset}/splits.json``](data/preprocessed/) like in [``./data/preprocessed/dblp/toy.dblp.v12.json/splits.json``](./data/preprocessed/dblp/toy.dblp.v12.json/splits.json)

**Model Architecture**
***Graph Neural Network***
We employ graph convolutional network to capture the underlying relations between experts, skills and locations. We construct a heterogeneous graph with nodes experts, skills and locations with edges represented as expert->skill, expert->location. The hyperparameters used for training gcn can be set in [``./src/config_prepare_dataset.py](./src/config_prepare_dataset.py). 

**Metapath Integration**
To incorporate a deeper connection of relationships in which a graph takes a predefined path between experts, skills and locations, we implement metapath based vector generation. We utilize three different sets of metapaths: 
``["experts", "skills", "experts"]``
``["experts", "skills", "experts", "skills", "experts"]``
``["experts", "skills", "loc", "skills", "experts"]``

The walk length for each metapath and number of metapaths generation is handled by ``walk_length``
``num_walks_per_node`` which can be set in [``./src/metapath2vec.py](./src/metapath2vec.py)
 
***Neural Model***
Each model has been defined in [``./src/mdl/``](./src/mdl/) under an inheritance hierarchy. They override abstract functions for ``train``, ``test``, ``eval``, and ``plot`` steps. 
For example, for our feedforward baseline [``fnn``](./src/mdl/fnn.py), the model has been implemented in [``./src/mdl/fnn.py``](src/mdl/fnn.py). Model's hyperparameters such as the learning rate (``lr``) or the number of epochs (``e``) can be set in [``./src/param.py``](src/param.py).

<p align="center"><img src='./src/mdl/model_inheritance_hierarchy.png' width="350" ></p>
  
Currently, we support neural models:
1) Bayesian [``bnn``](./src/mdl/bnn.py) where model's parameter (weights) is assumed to be drawn from Gaussian (Normal) distribution and the task is to not to learn the weight but the mean (μ) and standard deviation (σ) of the distribution at each parameter.
2) non-Bayesian feedforward [``fnn``](./src/mdl/fnn.py) where the model's parameter (weights) is to be learnt.

The input to the models is the vector representations for skills and the output is the vector representation for members. In another word, given the input skills, the models predict the members from the pool of candidates. We support two vector representations:
1) Sparse vector representation (occurrence or boolean vector): See preprocessing section above.
2) Dense vector representation ([``team2vec``](./src/mdl/team2vec.py)): Inspired by paragraph vectors by [Le and Mikolov](https://cs.stanford.edu/~quocle/paragraph_vector.pdf), we consider a team as a document and skills as the document words (``embtype == 'skill'``). Using distributed memory model, we map skills into a real-valued embedding space. Likewise and separately, we consider members as the document words and map members into real-valued vectors (``embtype == 'member'``). We also consider mapping skills and members into the same embedding space (``embtype == 'joint'``). Our embedding method benefits from [``gensim``](https://radimrehurek.com/gensim/) library.

**Negative Sampling Strategies**

As known, employing ``unsuccessful`` teams convey complementary negative signals to the model to alleviate the long-tail problem. Most real-world training datasets in the team formation domain, however, do not have explicit unsuccessful teams (e.g., collections of rejected papers.) In the absence of unsuccessful training instances, we proposed negative sampling strategies based on the ``closed-world`` assumption where no currently known successful group of experts for the required skills is assumed to be unsuccessful.  We study the effect of ``three`` different negative sampling strategies: two based on static distributions, and one based on adaptive noise distribution:

1) Uniform distribution (``uniform``), where subsets of experts are randomly chosen with the ``same probability`` as unsuccessful teams from the uniform distribution over all subsets of experts.

2) Unigram distribution (``unigram``), where subsets of experts are chosen regarding ``their frequency`` in all previous successful teams. Intuitively, teams of experts that have been more successful but for other skill subsets will be given a higher probability and chosen more frequently as a negative sample to dampen the effect of popularity bias.

3) Smoothed unigram distribution in each training minibatch (``unigram_b``), where we employed the ``add-1 or Laplace smoothing`` when computing the unigram distribution of the experts but in each training minibatch. Minibatch stochastic gradient descent is the _de facto_ method for neural models where the data is split into batches of data, each of which is sent to the model for the partial calculation to speed up training while maintaining high accuracy. 

To include a negative sampling strategy, there are two parameters for a model to set in [``./src/param.py``](src/param.py):
- ``ns``: the negative sampling strategy which can be ``uniform``, ``unigram``, ``unigram_b`` or ``None``(no negative sampling).
- ``nns``: number of negative samples

**Run**

The pipeline accepts three required list of values:
1) ``-data``: list of path to the raw datafiles, e.g., ``-data ./../data/raw/uspt/patent.tsv``
2) ``-domain``: list of domains of the raw data files that could be ``uspt`` or ``dblp``; e.g., ``-domain uspt``.
3) ``-model``: list of graph models that could be ``bnn_emb_gnn_meta``, ``bnn_emb_gnn_loc_meta``, ``bnn``, ``bnn_emb_gnn``, ``bnn_emb_gnn_loc``; e.g., ``-model bnn_emb_gnn_loc`` 

## 4. Results

We used [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval) to evaluate the performance of models on the test set as well as on their own train sets (should overfit) and validation sets. We report the predictions, evaluation metrics on each test instance, and average on all test instances in ``./output/{dataset name}/{model name}/{model's running setting}/``.  For example:

1) ``f0.test.pred`` is the predictions per test instance for a model which is trained folds [1,2,3,4] and validated on fold [0].
2) ``f0.test.pred.eval.csv`` is the values of evaluation metrics for the predictions per test instance
3) ``f0.test.pred.eval.mean.csv`` is the average of values for evaluation metrics over all test instances.
4) ``test.pred.eval.mean.csv`` is the average of values for evaluation metrics over all _n_ fold models.

**Benchmarks at Scale**

||min. #member's team: 75, min team size: 3, epochs: 20, learning rate: 0.1, hidden layer: [1, 100d], minibatch: 4096, #negative samples: 3|
|--------|------|
|Datasets|[uspt](https://patentsview.org/download/data-download-tables)|
|Metrics|recall@2,5,10, map@2,5,10, ndcg@2,5,10, p@2,5,10, auc|
|Baselines|{fnn,bnn}×{sparse, emb}×{none, uniform, unigram, unigram_b}|
|Results|[``./output/patent.tsv.filtered.mt75.ts3/``](./output/patent.tsv.filtered.mt75.ts3/)|

<p align="center">
<img src='https://user-images.githubusercontent.com/8619934/154041216-c80cccfb-70a2-4831-8781-cdb4718fb00e.png' >
<p align="center">
<img src='https://user-images.githubusercontent.com/8619934/154041087-e4d99b1e-eb6b-456a-837b-840e4bd5090a.png' >

Full predictions of all models on test and training sets and the values of evaluation metrics, per instance and average, are available in a rar file of size ``74.8GB`` and will be delivered upon request! 

## 5. Acknowledgement:
We benefit from [``pytrec_eval``](https://github.com/cvangysel/pytrec_eval), [``gensim``](https://radimrehurek.com/gensim/), [Josh Feldman's blog](https://joshfeldman.net/WeightUncertainty/), and other libraries. We would like to thank the authors of these libraries and helpful resources.
  
## 6. License:
©2021. This work is licensed under a [CC BY-NC-SA 4.0](license.txt) license.

## 7. Citation:
```
@inproceedings{DBLP:conf/cikm/DashtiSF22,
  author    = {Arman Dashti and Saeed Samet and Hossein Fani},
  title     = {Effective Neural Team Formation via Negative Samples},
  booktitle = {{CIKM} '22: The 31st {ACM} International Conference on Information and Knowledge Management, Hybrid Conference, Atlanta, Georgia, USA, October 17-21, 2022},
  publisher = {{ACM}},
  year      = {2022},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
  ```
  ```
@inproceedings{DBLP:conf/cikm/DashtiSPF22,
  author    = {Arman Dashti and Karan Saxena and Dhwani Patel and Hossein Fani},
  title     = {OpeNTF: A Benchmark Library for Neural Team Formation},
  booktitle = {{CIKM} '22: The 31st {ACM} International Conference on Information and Knowledge Management, Hybrid Conference, Atlanta, Georgia, USA, October 17-21, 2022},
  publisher = {{ACM}},
  year      = {2022},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
## 8. Awards:

> [CAD$300, Gold medalist, UWill Discover, 2022](https://scholar.uwindsor.ca/uwilldiscover/2022/2022Day3/30/)

> CAD$300, Best Research, Demo Day, School of Computer Science, University of Windsor, 2022. 
