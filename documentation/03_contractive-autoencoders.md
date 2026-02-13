# Contractive Auto-Encoders

> Explicit Invariance During Feature Extraction

## Introduction

Most Deep Learning Algorithms

- Basic building block of **feature extraction**
- Representation learned at one level is used as input for learning the next level
- Objective: These representations get better as depth is increased.

But

### **What defines a good representation?**

- PCA and ICA: working well studied
- RBMs, Sparse coding, Semi-supervised embedding: Not well understood yet
  - All of these produce non-linear representation which (unlike PCA,ICA) can be stacked (composed) to yield **deeper** levels of representation.
  - Deeper ===> Captures more abstract features

## Contribution

**What principles should guide the learning of such intermediate representations?**

- Capture as much info as possible (Aim of AE, Sparse AE)
- Representation must be useful in characterizing input distribution (Achieved by RBMs)

This paper introduces: **Penalty term**

- Can be used in both of the above contexts
- This penalty encourages the intermediate representation to be robust to small changes of the input around the training examples.

When combined with reconstruction error or likelihood criterion:

- Invariance obtained in the directions that make sense in the context of the given training data
- Variations present in the data should also be captured in the learned representations, but the other directions may be contracted in the learned representation

## How to extract robust features?

To encourage robustness of the representation $f(x)$ obtained for training input $x$, we propose to **penalize  it's sensitivity** to that input ==> **Frobenius Norm of the Jacobian $J_f(x)$ of the non-linear mapping.

Formally,

If input $x \in R^{d_x}$ is mapped by encoding function to hidden representation $h \in R^{d_h}$, this **sensitivity penalization term** is the **sum of squares of all partial derivatives of the extracted features w.r.t. to input dimensions**

$$
|| J_f (x) ||^2_F = \sum_{ij} (\frac {\partial h_j (x)}{\partial x_i)})^2
$$

Penalizing $||J_f||^2_F$ encourages 

- mapping of the feature space to be **contractive** in the neighborhood of the training data.

The flatness induced by having low valued first derivatives will imply an **invariance** or **robustness** for small variations of the input.

Thus **invariance** <==> **insensitivity** <==> **robustness** <==> **flatness** <==> **contraction**

Just the Jacobian term alone would encourage mapping to a useless constant representation.

- Thus it is counterbalanced in the autoencoder training by the need for the learned representation to allow a good reconstruction of the input

## Autoencoder Variants

In simplest form, any Auto-Encoder has 2 parts

- Encoder
- Decoder

First used as technique for dimensionality reduction:

- Where output of decoder represents the reduced representation.
- And the decoder is tuned to reconstruct original input from the encoder's representation through the minimization of a cost function.

More specifically

- When encoding activation functions are linear and number of hidden units is inferior to the input dimension (forming a bottle-neck), it has been shown
  - Learnt parameters of the encoder are a **SUB-SPACE** of the principal components of the input space (**PCA**)
- When encoding activation functions are non-linear, the AE can be expected to learn *more useful* feature detectors than what can be obtained using simple PCA.

Contrary to their original use of dimensionality reduction:

- The AE are often employed as **OVER-COMPLETE** (hidden units > input dim) setting to extract a number of features larger than the input dimension, yielding a *rich high dim representation*.
- In these cases, some form of regularization needed to avoid *uninteresting solutions* where autoencoder could perfectly reconstruct input without needing to extract any useful feature.

### Basic Autoencoder

- Encoder: Function $f$ that maps $x \in R^{d_x}$ to hidden representation $h(x) \in R^{d_h}$

$$
h = f(x) = s_f(Wx + b_h)
$$

Where,

- $s_f$ is non-linear activation function. (Typically sigmoid)

Encoder is parameterized by $d_h \times d_x$ weight matrix W and a bias vector $b_h \in R^{d_h}$.

- Decoder function $g$ maps hidden representation $h$ back to a reconstruction $y$

$$
y = g(h) = s_g(W'h + b_y)
$$

Where,

- $s_g$ is the decoder's activation function (identity (linear mapping) or sigmoid).

Decoder is parameterized by bias vector $b_y \in R^{d_x}$ and matrix W'

For simplicity we will assume $W' = W^T$

Autoencoder training consists of

- Finding params $\theta =   W, b_h, b_y $ that minimize the reconstruction error on a training set of $D_n$ examples

$$
J_{AE} (\theta) = \sum_{x \in {D_n}} L(x, g(f(x)))
$$

where L is the reconstruction error

- Typically it's the squared error (for linear reconstruction)
- Cross Entropy when activation function is Sigmoid and inputs are in [0,1]

### Regularized Autoencoders

Simplest form of regularization is **Weight Decay** which factors small weights by optimizing the following objective

$$
J_{AE+wd} (\theta) = \sum_{x \in {D_n}} L(x, g(f(x))) + \lambda \sum_{ij} W_{ij}^2
$$

where $ \lambda $ hyperparameter controls strength of regularization.

> Rather than having a *prior* on what the weights should be, it is possible to have a prior on what the hidden activations should be, Using this, several techniques have been developed to encourage **sparsity** in the representation 

### Denoising Autoencoders

Successful alternative form of regularization 

- Corrupt input before sending it through the autoencoder, that is trained to reconstruct clean version (i.e denoise)

$$
J_{DAE} (\theta) = \sum_{x \in D_n} E_{\tilde x \in q(\tilde x | x) [L(x, g(f(\tilde x)))]}
$$

where the expectation is over the corrupted versions $\tilde x$ of the examples x obtained from a corruption process $q(\tilde x | x)$

- Objective optimized by SGD (sampling corrupted examples)

Typically

- Consider corruptions as *additive isotropic Gaussian noise* $\tilde x = x + \epsilon$, $\epsilon \in N(0, \sigma^2I)$
- Binary masking noise where a fraction $v$ of input components (randomly chosen) have their value set to 0

Degree of corruption $\sigma$ or $v$ controls the degree of regularization.

## Contractive Auto-Encoders

**Motivation**: Robustness to small perturbations around the training points.

This is an alternative regularization that *favors mappings* that are more strongly **contracting** at the training samples.

The CAE is obtained with:

$$

J_{CAE} (\theta) = \sum_{x \in D_n} ( L(x, g(f(x))) + \lambda || J_f (x) ||_F^2)

$$

### Relationship with Weight Decay

- Frobenius Norm of Jacobian corresponds to **L2 weight decay** in the case of linear encoder (activation function is *identity*)
  - Here $J_{CAE} = J_{AE + wd}$

> In the linear case, keeping weights small is the only way to have a contraction

With **Sigmoid**, Contraction and Robustness can also be achieved by driving the hidden units to their saturated regime.

### Relationship with Sparse Auto-Encoders

Any AE that encourages sparse representations aim at having

- A majority of components of the representation close to zero
- They must have been computed in the **left** saturated part of the sigmoid non-linearity which is almost flat, with a tiny first derivative.

Thus SAE that outputs many close-to-zero features are likely to correspond  to a **highly contractive mapping** even though contraction/robustness is not explicitly encouraged via learning criterion.

### Relationship with Denoising Autoencoder

- Robustness  to input perturbations was also one of the motivation of the denoising auto-encoder.

Differences between DAE and CAE are:

- CAE explicitly encourage robustness of representation $f(x)$ whereas DAE encourages robustness of reconstruction $g o f(x)$ (which may only partially and indirectly encourage robustness of the representation, as the invariance requirement is shared between parts of AE)

This makes CAE better choice than DAE to learn useful feature extractors.

Since we only use *encoder* part for classification, robustness of extracted features appear more important than robustness of reconstruction

- DAE robustness is obtained stochastically by having several explicitly corrupted versions of training point aim for an identical reconstruction.
- CAE robustness to tiny perturbations is obtained **analytically** by penalizing the magnitude of first derivatives $|| J_f(x)||_F^2$ at training points only.
- 
> Analytic approximation for DAE's stochastic robustness criterion can be obtained in limit of very small additive Gaussian noise. This yields, not surprisingly, a term in $|| J_{gof}(x)||_F^2$ (Jacobian reconstruction) rather than the $|| J_f(x)||_F^2$ (Jacobian of representation) of CAEs.

#### Computation Considerations

In case of sigmoid non-linearity, penalty on Jacobian norm has the following simple expression:
$$
$|| J_f(x)||_F^2$ = \sum_{i=1}^{d_h} ( h_i (1 - h_i))^2 sum_{j=1}^{d_x} W_{ij}^2
$$

Computing this penalty (or its gradient) is similar and has about same cost as computing the overall construction error. Complexity equals $O(d_x \times d_h)$

## Experiments

Considered Models for unsupervised feature extraction:

- RBM-binary : Restricted Boltzmann Machine trained by Contrastive Divergence
- Basic AE
- AE + wd
- DAE + gaussian noise
- DAE + binary noise

All AE has
- tied weights (identical encoder, decoder)
- cross entropy reconstruction
- sigmoid activation

All gave out an encoder, later finetuned with MLP head for classification on **CIFAR grayscale** and **MNIST**

### Classification Performance

- 1 hidden layer of 1000 units, initialized with each of unsupervised algos under consideration
- Local Contraction Measure (average $||J_f||_F$ on the pretrained model strongly correlates with the final classification error.

CAE tries to explicitly minimize above measure and hence performs best.

- Stacking 2nd layer gave better performance, even better than SOTA 3 layer

### Closer Look into CONTRACTION

#### What happens locally: looking at the singular values of the Jacobian.

- A high dimensional Jacobian contains directional information:
   -  the amount of contraction is generally not the same in all directions.

This can be examined by SVD of $J_f$

#### What happens further away: contraction curves. 

The Frobenius norm of the Jacobian at some point x measures the contraction of the mapping locally at that point.

Contraction induced by proposed penalty can be measured beyond immediate training examples

-  ratio of the distances between two points in their original (input) space and their distance once mapped in the feature space.

This is called isotropic measure contraction ratio.

In the limit where the variation in the input space is infinitesimal, this corresponds to the derivative (i.e. Jacobian) of the representation map.

- For any encoding function $f$, we can measure average contraction ratio for pairs of points, one of which $x_0$ is picked from validation set and the other $x_1$ is randomly generated on a sphere of radius r centered on $x_0$ in input space. 

### Local Space Contraction

From geometrical POV,
- The robustness of the features can be seen as a contraction of the input space when projected in the feature space, in particular in the neighborhood of the examples from the data generating distribution.
- Otherwise (if the contraction was the same at all distances) it would not be useful, because it would just be a global scaling.

Thus this contraction happens with the proposed penalty, but much less without it

For all models except CAE and DAE-gaussian

- For DAE-g, the contraction ratio decreases (towards more contraction) as we move away from training samples (this is due to more saturation, and was expected).
- For CAE, the contraction ratio initially increases, up to the point where the effect of saturation takes over (the bump occurs at about the maximum distance between two training examples).

This can also be thought of the case where the training examples congregate near a low-dimensional manifold. The variations in data (translation/rotation) correspond to local dimensions along the manifold,  while the variations that are small or rare in the data correspond to the directions orthogonal to the manifold.

The proposed criterion:

-  Is trying to make the features invariant in all directions around the training examples.

But the reconstruction error (likelihood):

- Is making sure that that the representation is faithful, i.e., can be used to reconstruct the input example.

Hence the directions that resist to this contracting pressure (strong invariance to input changes) are the directions present in the training set.

> if the variations along these directions present in the training set were not preserved, neighboring training examples could not be distinguished and properly reconstructed.

- Hence the directions where the contraction is strong (small ratio, small singular values of the Jacobian matrix) are also the directions where the model believes that the input density drops quickly.
- Whereas the directions where the contraction is weak (closer to 1, larger contraction ratio, larger singular values of the Jacobian matrix) correspond to the directions where the model believes that the input density is flat (and large, since we are near a training example).

This contraction penalty thus helps the learner carve a kind of mountain supported by the training examples, and generalizing to a ridge between them.

What we would like is for these ridges to correspond to some directions of variation present in the data, associated with underlying factors of variation.

#### How far do these ridges extend around each training example and how flat are they? 

This can be visualized comparatively with the contraction ratio for different distances from the training examples.

> Different Features (elements of representation vector) ==> Ridges (direction of invariance) in different directions
>
> Dimensionality of these ridges gives a hint as to the local dimensionality of the manifold near which the data examples congregate. 

- The singular value spectrum of the Jacobian informs us about that geometry.
- The number of large singular values should reveal the dimensionality of these ridges, i.e., of that manifold near which examples concentrate. 

 The CAE by far does the best job at representing the data variations near a lower dimensional manifold, and the DAE is second best, while ordinary auto-encoders (regularized or not) do not succeed at all in this respect.

#### What happens when we stack a CAE on top of another one, to build a deeper encoder? 

Composing two CAEs yields even more contraction and even more non-linearity, i.e. a sharper profile, with a flatter level of contraction at short and medium distances, and a delayed effect of saturation (the bump only comes up at farther distances). 

## Conclusion

What makes a good representation:

- Besides being useful for a particular task, which we can measure, or towards which we can train a representation, this paper highlights the advantages for representations to be locally invariant in many directions of change of the raw input. 
- This idea is implemented by a penalty on the Frobenius norm of the Jacobian matrix of the encoder mapping, which computes the representation.
- The paper also introduces empirical measures of robustness and invariance, based on the contraction ratio of the learned mapping, at different distances and in different directions around the training examples.
- We hypothesize that this reveals the manifold structure learned by the model, and we find (by looking at the singular value spectrum of the mapping) that the Contractive Auto-Encoder discovers lower dimensional manifolds. 
- n addition, experiments on many datasets suggest that this penalty always helps an auto-encoder to perform better, and competes or improves upon the representations learned by Denoising Auto-Encoders or RBMs, in terms of classification error.