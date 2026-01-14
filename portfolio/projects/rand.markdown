---
layout: project
title: Anomalous Node Detection using RAND.
description: RAND - Reinforcement Neighborhood Selection for Unsupervised Graph Anomaly Detection
image: /assets/images/rand.png
github_link: https://github.com/Pietro-D/Anomalous-Node-Detection-using-RAND
tools: [Python, RL, Graph Representation, Anomaly Detection]
permalink: /projects/rand/
featured: true
---

# Table of Contents
* TOC
{:toc}
## Introduction

Today we explore [RAND](https://arxiv.org/pdf/2312.05526), a Reinforcement Learning framework
designed to detect anomalous nodes in graphs using an unsupervised approach. RAND stands out for several reasons: <br>
<ol>
<li>It achieves <b>state-of-the-art performance</b> when compared to other existing methods </li>
<li>It is theoretically proven to converge to the  optimal solution </li>
<li>Its computational complexity scales <b> quadratically </b> with the size of the graph</li>
</ol>
{: .stylish-list}
We will start with a  brief overview of the math ideas behind the method,
and then apply it to a graph dataset to see how it performs in practice.
In particular, we will experiment with a range of hyperparameter
configurations to undestand how different choices affect the results.<br>
<br>
As usual, the all the code used in this project is available on my GitHub repository, which you can find [here](https://github.com/Pietro-D/Anomalous-Node-Detection-using-RAND).

## How It Works
Formally, given a graph $$ G=(A,X) $$ with $$n$$ nodes,
where A is the adjacency matrix and X is the node attribute matrix, the RAND
framework can be divided into three main blocks.
### Reinforcement Neighborhood Selection
For each node the framework selects a neighborhood
using four different selection strategies: <br>
1) <b>1-hop neighbors</b> <br>
2) <b>2-hop neighbors</b> <br>
3) <b>K-Nearest Neighbors </b> <br>
4) <b> Personalized PageRank </b> <br>
At each iteration $$t$$, these strategies are combined probabilistically.<br>
Formally, for a given node $$i$$, the probability of being selected is defined as:<br>
<center>$$\Phi^t_i = \sum_k p^t_k \cdot Q^t_{k,i}$$ </center>
where $$ p^t_k$$ is the probability of using the $$k$$-th
selection strategy at iteration $$t$$, $$ Q^t_{k,i} $$ represents
the preference score of node $$i$$ under the $$k$$-th strategy.<br>
The probabilities $$p^t_k$$ are updated over time using RL.
The intuition behind the reward function is simple: a selection strategy
is considered better if it provides <b> coherent and homophilic</b> information,
meaning that connected nodes tend to exhibit similar anomaly behavior. <br>
To formalize this idea, for each node $$i$$ we define a vector: <br>
<center>$$C_i^t=softmax(\frac{1}{(y_i-y_j)+\epsilon}), j \in Neighbor(i)$$</center>
where $$y_j$$ is the current anomaly score of node $$j$$. <br>
The reward for the $$k$$-th selection strategy at iteration $$t$$ is then defined as:<br>
<center> $$r_k^t=\frac{1}{n}\sum_{i=1}^n C_i^t * \frac{p_k^t *Q_{k,N_i}}{\Phi_{N_i}^t}$$ </center>
This reward captures two key aspects:<br>
1) The distribution of the anomaly score differences among the neighbors of node $$i$$. <br>
2) The probability of selecting those neighbors using the $$k$$-th strategy. <br>

Using this reward, the weight of each strategy is updated according to: <br>
<center> $$w_k^{t+1} = w_k^t e^{ \frac{p_{\text{min}}}{2} \left( r_k + \frac{1}{p_k^t} \right) \delta_1 \sqrt{ \ln \left( \frac{n}{\delta_2} \right) /KT}}$$ </center>
where:<br>
1) $$\delta_1, \delta_2$$ are hyperparameters that control how aggressively the weights are updated.<br>
2) $$K$$ is the number of selection strategies.<br>
3) $$T$$ determines how frequently the probabilities $$p^t_k$$ are updated.<br>
Finally, the weights are converted into probabilities through normalization, while enforcing a lower bound $$p_{min}$$ (user-defined). This prevents any selection strategy from collapsing to zero probability and ensures continued exploration throughout training.
### Message Aggregator
Once the neighborhood of a given node is selected, the next step is to
aggregate information from its neighbors.
However, before performing this aggregation, potentially anomalous nodes
are masked.
This prevents them from aggregating information
from many normal nodes, which would make anomalies harder to detect.
To achieve this, an anchor representation $$E_{ca}$$ is defined to
capture the global structure of the graph, using an aggregation operator
such as <b>pooling</b>.
Next, nodes are ranked according to their distance from the anchor.
The top $$mr$$ fraction of nodes with the largest distance form $$E_{ca}$$
are maskered. Formally:<br>
<center> $$M_{id}=Top_{(|V|\cdot mr)}[argsort_{max}(||E_i-E_{ca}||_2^2), i \in V]$$</center>
For the masked nodes, their representations remain unchanged:<br>
<center>$$H_i=E_i \quad i \in M_{id}$$</center>
For the remaining (unmasked) nodes, <b>attention</b> is used to aggregate
information from their neighbors:<br>
<center>$$att_i=Tanh( \;Concat( \;[E_i, \; [E_j, j \in Neighbor(i)] \; ] \; \cdot W_{att}))$$</center>
<center>$$H_i= (att_{i,i} \: \odot \: E_i + \sum_{j \in N_i} att_{i,j} \: \odot \: E_j)W_{mp}$$</center>

### Anomaly Scores
The final step consists of computing an <b>anomaly score</b> for each node:<br>
<center>$$ score(i)= \alpha E_{attr,i} + (1-\alpha)E_{topo,i}$$</center>
As shown in the equation above, the score balances, through the parameter
$$\alpha$$, two different types of errors: <br>
1) <b> Topology reconstruction error</b> <br>
2) <b> Attribute reconstruction error</b> <br>
The underlying intuition is that anomalous nodes are harder reconstruct,
both in terms of their structural role in the graph and their attributes.<br>
The <b> topology reconstruction error </b> is
defined as the distance between the original adjacency matrix and its
reconstructed version: <br>
<center>$$ \hat{A}= H \cdot H^T$$</center>
<center> $$  E_{topo}= ||A-\hat{A}||_2^2 \in \mathbb{R}^n$$ </center>
Similarly, the <b>attribute reconstruction error</b> measures
the distance between the original attribute matrix $$X$$
and its reconstruction $$\hat{X}$$. <br>
<center>$$\hat{X}= graph_{encoding}(H,A)$$</center>
<center>$$  E_{attr}= ||X-\hat{X}||_2^2 \in \mathbb{R}^n$$</center>
The reconstructed attributes are obtained 
using a graph encoder, like an <b>MLP</b> or a <b>GCN</b>.

## Experiments
To evaluate the model, we use the <b>AmazonRatingsDataset</b>
from the <b>DGL</b> library.<br>
Since this dataset does not contain ground-truth anomalies,
we inject them artificially using two different techniques: <br>
1) <b>Topological anomalies</b>: These are created by adding small
random cliques to the graph. The intuition is that a tightly connected
group of nodes can be considered anomalous due to its unusually high
connectivity compared to the rest of the graph <br>
2)<b>Contextual anomalies</b>: In this case, we randomly select nodes
and perturb their attributes. The idea is that an anomalous node will
exhibit attribute values that significantly differ from those
of its nighborhood.<br>
The table below summarizes some statistics of the resulting graph after anomaly injection :<br>


|  | NUM. |
|:-------------|:---------------|
| Nodes        | 24.492           |
| Edges        | 190.300           |
| Features     | 300            |
| Topological Anomalies  | 300 |
| Contextual Anomalies   |    300     |
{: .tech-table}
<div class="table-caption">Table 1: AmazonRatingsDataset after anomaly injection </div>

As a first experiment, we vary the value of $$\alpha$$, which controls the balance between
topological and attribute reconstruction errors. The results are reported
below: <br>


|  | $$\alpha=0.2$$ | $$\alpha=0.4$$ | $$\alpha=0.6$$ | $$\alpha=0.8$$ |
|:-------------|:---------------|:---------------|:---------------|:---------------|
| Topological AUC        | **0.97** | 0.96 | 0.95     | 0.87     |
| Attribute AUC          | 0.74     | 0.85 | 0.93     | **0.97** |
| Total AUC              | 0.85     | 0.90 | **0.94** | 0.92     |
| Topological AP         | **0.15** | 0.14 | 0.11     | 0.05     |
| Attribute AP           | 0.02     | 0.04 | 0.11     | **0.29** |
| Total AP               | 0.12     | 0.15 | 0.20     | **0.24** |
{: .tech-table}
<div class="table-caption">Table 2 </div>
As expected, when $$\alpha<0.5$$ more importance is given to the
<b> topological reconstruction error </b>, resulting in better performance on topology-related metrics.
However, the best overall performance is achieved for values of $$\alpha$$
greater than $$0.5$$, where attribute information plays a more significant role.

Next we experiment with different values of $$p_{min}$$ which sets a
<b>lower bound</b> on the strategy probabilities $$p_k$$.
As shown in the table below, this parameter doesn't significantly
affect either AUC or AP metrics:<br>

|  | $$p_{min}=0.0$$ | $$p_{min}=0.05$$ | $$p_{min}=0.1$$ | $$p_{min}=0.2$$ |
|:-------------|:---------------|:---------------|:---------------|:---------------|
| Topological AUC        | 0.87     | 0.87 | 0.80     | 0.87     |
| Attribute AUC          | 0.97     | 0.97 | 0.97     | 0.97     |
| Total AUC              | 0.92     | 0.92 | 0.88     | 0.88     |
| Topological AP         | 0.04     | 0.05 | 0.03     | 0.03     |
| Attribute AP           | 0.29     | 0.29 | 0.31     | 0.31     |
| Total AP               | 0.24     | 0.24 | 0.23     | 0.23     |
{: .tech-table}
<div class="table-caption">Table 3 </div>
Interestingly, while performance metrics remain largely unchanged, the
distribution of the final strategy probabilities varies significantly
with different values of $$p_{min}$$. In particular, we observe the following behaviors:<br>

1) When $$p_{min}=0$$, all four strategies converge to the same probability ($$0.25$$).<br>
2) As $$p_{min} \rightarrow 0$$, one selection strategy tends to become dominant,
receiving much higher probability than the others.<br>
3) As $$p_{min}$$  approaches its upper bound (0.25 when using four
strategies), the probability distribution becomes increasingly
uniform.<br>
<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" style="max-width: 600px; height: auto;">
  <!-- Background -->
  <rect width="800" height="600" fill="white"/>
  
  <!-- Grid lines -->
  <line x1="100" y1="500" x2="700" y2="500" stroke="#ddd" stroke-width="1"/>
  <line x1="100" y1="400" x2="700" y2="400" stroke="#ddd" stroke-width="1"/>
  <line x1="100" y1="300" x2="700" y2="300" stroke="#ddd" stroke-width="1"/>
  <line x1="100" y1="200" x2="700" y2="200" stroke="#ddd" stroke-width="1"/>
  <line x1="100" y1="100" x2="700" y2="100" stroke="#ddd" stroke-width="1"/>
  
  <!-- Axes -->
  <line x1="100" y1="500" x2="700" y2="500" stroke="black" stroke-width="2"/>
  <line x1="100" y1="500" x2="100" y2="80" stroke="black" stroke-width="2"/>
  
  <!-- Y-axis labels -->
  <text x="70" y="505" font-family="Arial" font-size="14" text-anchor="end">0.0</text>
  <text x="70" y="405" font-family="Arial" font-size="14" text-anchor="end">0.2</text>
  <text x="70" y="305" font-family="Arial" font-size="14" text-anchor="end">0.4</text>
  <text x="70" y="205" font-family="Arial" font-size="14" text-anchor="end">0.6</text>
  <text x="70" y="105" font-family="Arial" font-size="14" text-anchor="end">0.8</text>
  <text x="70" y="85" font-family="Arial" font-size="14" text-anchor="end">1.0</text>
  
  <!-- Y-axis title -->
  <text x="30" y="290" font-family="Arial" font-size="16" text-anchor="middle" transform="rotate(-90 30 290)">Probability</text>
  
  <!-- X-axis title -->
  <text x="400" y="550" font-family="Arial" font-size="16" text-anchor="middle">Selection Strategies</text>
  
  <!-- PPR bars -->
  <rect x="140" y="400" width="30" height="100" fill="#7EB6E6"/>
  <rect x="172" y="460" width="30" height="40" fill="#F0A378"/>
  <rect x="204" y="480" width="30" height="20" fill="#90D690"/>
  <rect x="236" y="420" width="30" height="80" fill="#F09090"/>
  
  <!-- Hop 1 bars -->
  <rect x="310" y="400" width="30" height="100" fill="#7EB6E6"/>
  <rect x="342" y="220" width="30" height="280" fill="#F0A378"/>
  <rect x="374" y="212" width="30" height="288" fill="#90D690"/>
  <rect x="406" y="348" width="30" height="152" fill="#F09090"/>
  
  <!-- Hop 2 bars -->
  <rect x="480" y="400" width="30" height="100" fill="#7EB6E6"/>
  <rect x="512" y="460" width="30" height="40" fill="#F0A378"/>
  <rect x="544" y="480" width="30" height="20" fill="#90D690"/>
  <rect x="576" y="420" width="30" height="80" fill="#F09090"/>
  
  <!-- KNN bars -->
  <rect x="650" y="400" width="30" height="100" fill="#7EB6E6"/>
  <rect x="682" y="460" width="30" height="40" fill="#F0A378"/>
  <rect x="714" y="428" width="30" height="72" fill="#90D690"/>
  <rect x="746" y="412" width="30" height="88" fill="#F09090"/>
  
  <!-- X-axis labels -->
  <text x="205" y="525" font-family="Arial" font-size="14" text-anchor="middle">PPR</text>
  <text x="375" y="525" font-family="Arial" font-size="14" text-anchor="middle">Hop 1</text>
  <text x="545" y="525" font-family="Arial" font-size="14" text-anchor="middle">Hop 2</text>
  <text x="715" y="525" font-family="Arial" font-size="14" text-anchor="middle">KNN</text>
  
  <!-- Value labels -->
  <text x="155" y="390" font-family="Arial" font-size="11" text-anchor="middle">0.25</text>
  <text x="187" y="450" font-family="Arial" font-size="11" text-anchor="middle">0.10</text>
  <text x="219" y="470" font-family="Arial" font-size="11" text-anchor="middle">0.05</text>
  <text x="251" y="410" font-family="Arial" font-size="11" text-anchor="middle">0.20</text>
  
  <text x="325" y="390" font-family="Arial" font-size="11" text-anchor="middle">0.25</text>
  <text x="357" y="210" font-family="Arial" font-size="11" text-anchor="middle">0.70</text>
  <text x="389" y="200" font-family="Arial" font-size="11" text-anchor="middle">0.72</text>
  <text x="421" y="338" font-family="Arial" font-size="11" text-anchor="middle">0.38</text>
  
  <text x="495" y="390" font-family="Arial" font-size="11" text-anchor="middle">0.25</text>
  <text x="527" y="450" font-family="Arial" font-size="11" text-anchor="middle">0.10</text>
  <text x="559" y="470" font-family="Arial" font-size="11" text-anchor="middle">0.05</text>
  <text x="591" y="410" font-family="Arial" font-size="11" text-anchor="middle">0.20</text>
  
  <text x="665" y="390" font-family="Arial" font-size="11" text-anchor="middle">0.25</text>
  <text x="697" y="450" font-family="Arial" font-size="11" text-anchor="middle">0.10</text>
  <text x="729" y="418" font-family="Arial" font-size="11" text-anchor="middle">0.18</text>
  <text x="761" y="402" font-family="Arial" font-size="11" text-anchor="middle">0.22</text>
  
  <!-- Legend -->
  <rect x="570" y="30" width="210" height="100" fill="white" stroke="#ccc" stroke-width="1"/>
  <text x="675" y="50" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">Distribution</text>
  
  <rect x="580" y="60" width="20" height="12" fill="#7EB6E6"/>
  <text x="605" y="70" font-family="Arial" font-size="12">p_min=0</text>
  
  <rect x="580" y="77" width="20" height="12" fill="#F0A378"/>
  <text x="605" y="87" font-family="Arial" font-size="12">p_min=0.1</text>
  
  <rect x="580" y="94" width="20" height="12" fill="#90D690"/>
  <text x="605" y="104" font-family="Arial" font-size="12">p_min=0.05</text>
  
  <rect x="580" y="111" width="20" height="12" fill="#F09090"/>
  <text x="605" y="121" font-family="Arial" font-size="12">p_min=0.2</text>
</svg>

As a final experiment, we modify the set of selection strategies used
by the model.
First, we replace PPR with the <b> Katz Centrality </b>, which for
a given node $$i$$ is defined as:<br>
<center>$$C_{katz}(i)=\sum_{k=1}^K \beta^k(A^T)^k$$</center>
The parameter $$\beta<1$$, known as the <b>attenuation parameter</b>,
reduces the influence of nighbors as the distance from the node increases.
The parameter $$K$$ represents the maximun path length considered. In
our experiment, we set $$\beta=0.5$$ and $$K=10$$.<br>
Intuitively, a node with high katz centrality is not only connected to
many other nodes, but is also connected to <b>important nodes</b>
(i.e. nodes that are themselves highly central).<br>
Next, we replace the 1-hop neighborhood strategy with
<b> Jaccard similarity </b>, defined as:<br>
<center>$$ sim_{jaccard}(i,j)=\frac{|N(i) \cap N(j)|}{|N(i) \cup N(j)|}$$</center>
We trained this modified version of RAND using the default
hyperparameter configuration and we found performance
comparable to the results reported in Table 2.

## Conclusions
RAND is a powerful framework for anomaly detection in graphs,
and is particularly well suited to real-world scenarios where
no prior knowledge about anomalous nodes is available.
Through our experiments, we observe that the method is <b>robust</b> to
variantions in hyperparameters, showing stable results across a wide
range of configurations. Moreover, we find that modifying the selection
strategies does not significantly degrade performance, suggesting that
RAND is flexible and adaptable to different design choices.<br>

To conclude, we highlight some promising directions for future work:<br>
<ol>
  <li>Adapting RAND to multilayer or multiplex graphs</li>
  <li>Exploring alternative aggregation strategies, such as Multi-Head Attention.</li>
  <li>Developing explainability mechanisms to better interpret and understand the detected anomalies.</li>
</ol>
{: .stylish-list}
