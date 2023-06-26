# ``OpeNTF``: An Open-Source Neural Team Formation Benchmark Library 
Effective team formation is crucial for project success, influenced by skill diversity and geographic proximity. This paper proposes a novel approach for geo-spatial team formation that leverages graph neural network (gnn) embeddings to transfer knowledge from a heterogenous collaborative network among experts, with metapaths and lack thereof, followed by a neural-based multi-label classifier. Our method aims to optimize team composition by considering the interplay between skill compatibility and geographic cohesiveness. Specifically, we construct a heterogeneous graph representation whose nodes are experts, skills, and locations to capture the complex relationships between experts' skills and their geographic locations.. We employ gnn to learn vector representations of experts, encoding both their skill profiles and geographic information using metapaths. We propose a multi-objective optimization to guide the team formation process. The objective is to maximize skill diversity while minimizing geographic dispersion, balancing effective collaboration and efficient communication. We utilize classification and information retrieval metrics to evaluate the accuracy of the recommended teams of experts concerning the required skills and geographical distribution. Experimental evaluations of our proposed method on a real-world dataset of patents and computer science publications against baseline methods demonstrated the effectiveness of our approach in forming diverse and geographically cohesive teams. The findings of this study contribute to the field of team formation by highlighting the benefits of incorporating gnn embeddings considering skill and location in tandem for the task of team formation.

## :movie_camera: [`Demo`](https://youtu.be/nFLow5Oy9nw)
In this paper, we address the geo-spatial team formation problem; given a set of experts, complementary skills and locations, which includes experts’ geographical location in terms of country, province, or city, the goal is to find the optimal team and to figure out whether the combination of two characteristics such as skills and locations has a synergistic effect. We define an optimal team as one whose members are most likely to produce a desired output and are proven successful in their respective domains. Prior works have been shown to address the problem of team formation by using skills as a primary factor which affects a team’s performance[4] but overlooking location and the corresponding ties it leads to between experts within a team. Integrating geographic location is crucial in today's globalized work environment. Although remote work over online platforms is growing, geographical proximity remains important for face-to-face interactions, cultural understanding, time zone differences, trust and team cohesion and local expertise and resources. The team members' geographical location can significantly impact team dynamics, coordination, and effectiveness. By considering both skills and locations in team formation, organizations can strive to create teams that are both skill-diverse and geographically cohesive, promoting efficient collaboration and reducing potential challenges associated with distance and time zone differences. However, despite its importance, including location as a criterion in conjunction with skill in team formation has yet to be studied.
By understanding the relationship between skills and locations, organizations can strategically assemble teams that leverage expertise and geographic proximity, leading to improved communication, knowledge sharing, and potentially higher team performance. This research aims to bridge the gap in existing studies by investigating the combined influence of skills and locations on team formation and identifying the key factors contributing to successful team outcomes in a geospatial context. This research follows the recent trends in the team formation problem, which involves using neural networks, graph neural network approaches, testing our proposed methodology on different datasets, such as USPT[5] and DBLP[6], and comparing our results with the state-of-the-art baselines. We define terms used in our work, such as experts as individuals with their respective patents in the case of USPT and publications in the case of the DBLP dataset. Skills are the possible field in which the experts worked to get their respective patents/publications accepted. We refer to locations as experts’ geolocation at the time of their submission for a patent/publication. A successful expert is an expert who has their respective work accepted for their respective fields. We utilize a graph neural network by making a heterogeneous graph consisting of nodes of experts, skills and locations. We utilize two different approaches for embedding generation from graphs that include without metapath training and with metapath training. A metapath is a walk taken by a graph neural network from a source node to a destination node so that the trained model attains better coverage and captures neighbourhood information for that particular source node. These trained embeddings are then processed into teams where each team consists of skills, experts and location for a particular patent or a publication. We then feed these skills and locations of a team into a bayesian neural network whose output is the most probable set of experts.
 

1. [Setup](#1-setup)
2. [Quickstart](#2-quickstart)
3. [Features](#3-features)
4. [Results](#4-results)
5. [Acknowledgement](#5-acknowledgement)
6. [License](#6-license)
7. [Citation](#7-citation)
8. [Awards](#8-awards)

## 1. [Setup](https://colab.research.google.com/github/fani-lab/OpeNTF/blob/main/quickstart.ipynb)
You need to have ``Python >= 3.8`` and install the following main packages, among others listed in [``requirements.txt``](requirements.txt):
```
torch>=1.9.0
pytrec-eval-terrier==0.5.2
gensim==3.8.3
```
By ``pip``, clone the codebase and install required packages:
```sh
git clone https://github.com/Fani-Lab/opentf
cd opentf
pip install -r requirements.txt
```
By [``conda``](https://www.anaconda.com/products/individual):

```sh
git clone https://github.com/Fani-Lab/opentf
cd opentf
conda env create -f environment.yml
conda activate opentf
```

For installation of specific version of a python package due to, e.g., ``CUDA`` versions compatibility, one can edit [``requirements.txt``](requirements.txt) or [``environment.yml``](environment.yml) like as follows:

```
# CUDA 10.1
torch==1.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
## 2. Quickstart [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fani-lab/OpeNTF/blob/main/quickstart.ipynb)

```sh
cd src
python main.py -data ../data/raw/uspt/toy.patent.tsv -domain uspt -model bnn_emb_gnn -filter 0
```

The above run, loads and preprocesses a tiny-size toy example dataset [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z) followed by 5-fold train-evaluation on a training split and final test on the test set for ``feedforward`` and ``Bayesian`` neural models using default hyperparameters from [``./src/param.py``](./src/param.py).

## 3. Features
**Datasets and Parallel Preprocessing**

Raw dataset, e.g., scholarly papers from AMiner's citation network dataset of [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), movies from [``imdb``](https://datasets.imdbws.com/), or US patents from [``uspt``](https://patentsview.org/download/data-download-tables) were assumed to be populated in [``data/raw``](data/raw). For the sake of integration test, tiny-size toy example datasets [``toy.dblp.v12.json``](data/raw/dblp/toy.dblp.v12.json) from [``dblp``](https://originalstatic.aminer.cn/misc/dblp.v12.7z), [[``toy.title.basics.tsv``](data/raw/imdb/toy.title.basics.tsv), [``toy.title.principals.tsv``](data/raw/imdb/toy.title.principals.tsv), [``toy.name.basics.tsv``](data/raw/imdb/toy.name.basics.tsv)] from [``imdb``](https://datasets.imdbws.com/) and [``toy.patent.tsv``](data/preprocessed/uspt/toy.patent.tsv) have been already provided.

<p align="center"><img src='./src/cmn/team_inheritance_hierarchy.png' width="250" ></p>

Raw data will be preprocessed into two main ``sparse`` matrices each row of which represents: 

>i) ``vecs['member']``: occurrence (boolean) vector representation for members of a team, e.g., authors of a paper or crew members of a movie,
> 
>ii) ``vecs['skill']``: occurrence (boolean) vector representation for required skills for a team, e.g., keywords of a paper or genre of a movie.

Also, indexes will be created to map the vector's indexes to members' names and skills' names, i.e., ``i2c``, ``c2i``, ``i2s``, ``s2i``.

The sparse matrices and the indices will be persisted in [``data/preprocessed/{dblp,imdb,uspt}/{name of dataset}``](data/preprocessed/) as pickles ``teamsvecs.pkl`` and ``indexes.pkl``. For example, the preprocessed data for our dblp toy example are [``data/preprocessed/dblp/toy.dblp.v12.json/teamsvecs.pkl``](data/preprocessed/dblp/toy.dblp.v12.json/teams.pkl) and [``data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl``](data/preprocessed/dblp/toy.dblp.v12.json/indexes.pkl).

> Our pipeline benefits from parallel generation of sparse matrices for teams that significantly reduces the preprocessing time as shown below:
> 
> <p align="center"><img src="./data/speedup.jpg" width="200"><img src="./data/speedup_loglog.jpg" width="190"></p>


Please note that the preprocessing step will be executed once. Subsequent runs load the persisted pickle files. In order to regenerate them, one should simply delete them. 

**[Train][Validation][Test] Split**

We randomly take ``85%`` of the dataset for the train-validation set and ``15%`` as the test set, i.e., the model never sees these instances during training or model tuning. You can change ``train_test_split`` parameter in [``./src/param.py``](./src/param.py).

We use _n_-fold cross-validation, that is, we train a model _n_ times on _(n-1)_ folds and utilize the remaining fold as the validation set to adjust the learning rate during training. The number of folds is set by ``nfolds`` in [``./src/param.py``](./src/param.py). In total, we have _n_ models each of which will be evaluated on the test set.

At each run, we store ids of instances in train-validation folds and test set in [``./data/preprocessed/{dblp,imdb,uspt}/{name of dataset}/splits.json``](data/preprocessed/) like in [``./data/preprocessed/dblp/toy.dblp.v12.json/splits.json``](./data/preprocessed/dblp/toy.dblp.v12.json/splits.json)

**Model Architecture**

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
1) ``-data``: list of path to the raw datafiles, e.g., ``-data ./../data/raw/dblp/dblp.v12.json``, or the main file of a dataset, e.g., ``-data ./../data/raw/imdb/title.basics.tsv``
2) ``-domain``: list of domains of the raw data files that could be ``dblp``, ``imdb``, or `uspt`; e.g., ``-domain dblp imdb``.
3) ``-model``: list of baseline models that could be ``fnn``, ``fnn_emb``, ``bnn``, ``bnn_emb``, ``random``; e.g., ``-model random fnn bnn`` 

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
![image](https://github.com/fani-lab/OpeNTF/assets/19558771/792ab813-ce1a-4bc4-8a6b-9011621658ba)


![image](https://github.com/fani-lab/OpeNTF/assets/19558771/f6836aa1-7943-477d-acbc-a3ae60e94313)


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
