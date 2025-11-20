# Machine Learning

-  <mark>Simonyan 2015</mark>: Very deep convolutional neural networks for large scale image recognition 

	- In this work we investigate the **effect of the CNN depth on its accuracy** in the large-scale image recognition setting. Our main contribution is a thorough evaluation of networks of increasing depth using an architecture with very small (3 × 3) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to **16–19 weight layers**. 
	- These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisa- tion and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. We have made our two best-performing ConvNet models publicly available to facilitate further research on the use of deep visual representations in computer vision.

---

-  <mark>Gal 2016</mark>: Dropout as a Bayesian Approximation: Representing Model Uncertainty in Dep Learning

	- ML tools for regression and classification do not capture **model uncertainty**. In comparison, Bayesian models offer a mathematically grounded framework to reason about model uncertainty, but usually come with a prohibitive computational cost. In this paper we develop a new theoretical framework casting **dropout training in deep NNs as approximate Bayesian inference in deep Gaussian processes**. 
	- A direct result of this theory gives us **tools to model uncertainty** with dropout NNs – extracting information from existing models that has been thrown away so far. 
	- This mitigates the problem of representing uncertainty in deep learning without sacrificing either computational complexity or test accuracy. We perform an extensive study of the properties of dropout’s uncertainty. Various network architectures and nonlinearities are assessed on tasks of regression and classification, using MNIST as an example. 
	- We show a considerable improvement in predictive log-likelihood and RMSE compared to existing state-of-the-art methods, and finish by using dropout’s uncertainty in deep reinforcement learning.

---

- <mark>He 2016</mark>: Deep Residual Learning for Image Recognition

	- Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. 
	- On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers—8×deeper than **VGG** nets but still having lower complexity. An ensemble of these residual nets achieves $3.57\%$ error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. 
	- Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

---

- <mark>Krizhevsky 2012</mark>: ImageNet Classification with Deep Convolutional Neural Networks

	- We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 dif- ferent classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a final 1000-way softmax. To make train- ing faster, we used non-saturating neurons and a very efficient GPU implemen- tation of the convolution operation. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called “dropout” that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.

---

- <mark>Lucie-Smith 2018</mark>: Machine learning cosmological structure formation

	- We train a ML algorithm to learn cosmological structure formation from N-body simulations. The algorithm infers the **relationship between the initial conditions and the final dark matter haloes**, without the need to introduce approximate halo collapse models. 
	- We gain **insights into the physics driving halo formation** by evaluating the predictive performance of the algorithm when provided with different types of information about the local environment around dark matter particles. 
	- **The algorithm learns to predict whether or not dark matter particles will end up in haloes of a given mass range**, based on spherical overdensities. 
	- We show that the resulting predictions match those of spherical collapse approximations such as extended Press-Schechter theory. 
	- Additional information on the shape of the local gravitational potential is **not able to improve halo collapse predictions**; the linear density field contains sufficient information for the algorithm to also reproduce **ellipsoidal collapse** predictions based on the **Sheth-Tormen model**. 
	- We investigate the algorithm’s performance in terms of halo mass and radial position and perform **blind analyses on independent initial conditions realisations** to demonstrate the generality of our results.

---

- <mark>Szegedy 2014</mark>: Going deeper with convolutions

	- We propose a deep convolutional NN architecture codenamed Inception, which was responsible for setting the **new state of the art for classification and detection in the ImageNet Large-Scale Visual Recognition Challenge 2014** (ILSVRC14). 
	- The main hallmark of this architecture is the improved utilization of the computing resources inside the network. This was achieved by a carefully crafted design that allows for **increasing the depth and width of the network while keeping the computational budget constant**. 
	- To optimize quality, the architectural decisions were based on the **Hebbian principle** and the **intuition of multi-scale processing**. 
	- One particular incarnation used in our submission for ILSVRC14 is called **GoogLeNet**, a 22 layers deep network, the quality of which is assessed in the context of classification and detection.

---

- <mark>Dai 2018</mark>: Galaxy Morphology Classification with Deep Convolutional Neural Networks

	- We propose a variant of ResNets for **galaxy morphology classification**. The variant, together with other popular CNNs, are applied to a sample of **28790 galaxy images** from **Galaxy Zoo 2 dataset**, to classify galaxies into **five classes**, i.e. completely round smooth, in-between smooth (between completely round and cigar-shaped), cigar-shaped smooth, edge-on and spiral. A variety of metrics, such as accuracy, precision, recall, F1 value and AUC, show that the proposed network achieves the **state-of-the-art classification performance** among the networks, namely, Dieleman, AlexNet, VGG, Inception and ResNets. 
	- The overall classification accuracy of our network on the testing set is **95.2083%** and the accuracy of each type is given as: completely round, 96.6785%; in-between, 94.4238%; cigar-shaped, 58.6207%; edge-on, 94.3590% and spiral, 97.6953% respectively. 
	- Our model algorithm can be applied to large-scale galaxy classification in forthcoming surveys such as the Large Synoptic Survey Telescope (LSST)
	- Comments: 
		- Is it openly available? yes, code [Github](https://github.com/Adaydl/GalaxyClassification/tree/master) and data [Kaggle](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge)
		- Their preprocessing pipeline and training seems smart

---

- <mark>De la Calleja 2004</mark>: Machine learning and image analysis for morphological galaxy classification

	- We present an experimental study of ML and image analysis for performing automated morphological galaxy classification. We used a **NN**, and a **locally weighted regression** method, and implemented **homogeneous ensembles of classifiers**. The ensemble of NNs was created using the **bagging ensemble method**, and manipulation of input features was used to create the ensemble of locally weighed regression. 
	- The galaxies used were rotated, centred, and cropped, all in a fully automatic manner. In addition, we used **PCA** to reduce the dimensionality of the data, and to extract relevant information in the images. 
	- Preliminary experimental results using 10-fold CV show that the homogeneous ensemble of locally weighted regression produces the best results, with over $91\%$ accuracy when considering three galaxy types (E, S and Irr), and over $95\%$ accuracy for two types (E and S).

---

- <mark>Dieleman 2015</mark>: Rotation-invariant convolutional neural networks for galaxy morphology prediction

	- Measuring the morphological parameters of galaxies is a key requirement for studying their formation and evolution. Surveys such as the Sloan Digital Sky Survey (SDSS) have resulted in the availability of very large collections of images, which have permitted population-wide analyses of galaxy morphology. 
	- Morphological analysis has traditionally been carried out mostly via **visual inspection by trained experts**, which is time consuming and does not scale to large   ($\gtrsim10^4$) numbers of images. Although attempts have been made to build automated classification systems, these have **not been able to achieve the desired level of accuracy**. 
	- The Galaxy Zoo project successfully applied a **crowdsourcing strategy**, inviting online users to classify images by answering a series of questions. Unfortunately, even this approach **does not scale well enough** to keep up with the increasing availability of galaxy images. 
	- We present a deep NN model for galaxy morphology classification which exploits **translational and rotational symmetry**. It was developed in the context of the **Galaxy Challenge**, an international competition to build the best model for morphology classification based on annotated images from the Galaxy Zoo project. For images with high agreement among the Galaxy Zoo participants, our model is able to reproduce their consensus with **near-perfect accuracy** ($>99$) for most questions. Confident model predictions are highly accurate, which makes the model suitable for filtering large collections of images and forwarding challenging images to experts for manual annotation. 
	- This approach greatly reduces the experts’ workload without affecting accuracy. The application of these algorithms to larger sets of training data will be critical for analysing results from future surveys such as the Large Synoptic Survey Telescope

---

- <mark>Chen 2020</mark>: Concept Whitening for Interpretable Image Recognition

	- What does a NN encode about a concept as we traverse through the layers? Interpretability in ML is undoubtedly important, but the calculations of neural networks are very challenging to understand. Attempts to see inside their hidden layers can either be **misleading, unusable, or rely on the latent space to possess properties that it may not have**. 
	- In this work, rather than attempting to analyze a NN posthoc, we introduce a mechanism, called **concept whitening** (CW), to alter a given layer of the network to allow us to better understand the computation leading up to that layer. When a concept whitening module is added to a CNN, **the axes of the latent space are aligned with known concepts of interest**. 
	- By experiment, we show that CW can provide us a much clearer understanding for how the network gradually learns concepts over layers. CW is an alternative to a **batch normalization layer** in that it normalizes, and also **decorrelates** (whitens) the latent space. 
	- CW can be used in any layer of the network without hurting predictive performance.

---

- <mark>Katebi 2018</mark>: Galaxy morphology prediction using capsule networks

	- Understanding morphological types of galaxies is a key parameter for studying their formation and evolution. NN that have been used previously for galaxy morphology classification have some disadvantages, such as **not being invariant under rotation**.
	- In this work, we studied the performance of Capsule Network, a recently introduced NN architecture that is rotationally invariant and spatially aware, on the task of galaxy morphology classification. We designed two evaluation scenarios based on the answers from the question tree in the **Galaxy Zoo project**. 
		- In the first scenario, we used Capsule Network for regression and predicted probabilities for all of the questions. 
		- In the second scenario, we chose the answer to the first morphology question that had the highest user agreement as the class of the object and trained a Capsule Network classifier, where we also **reconstructed galaxy images**. 
	- We achieved promising results in both of these scenarios. Automated approaches such as the one introduced here will greatly decrease the workload of astronomers and will play a critical role in the upcoming large sky surveys.

---

- <mark>Cavanagh 2021</mark>: Morphological classification of galaxies with deep learning: comparing 3-way and 4-way CNNs

	- Classifying the morphologies of galaxies is an important step in understanding their physical properties and evolutionary histories. The advent of large-scale surveys has hastened the need to develop techniques for automated morphological classification. 
	- We train and test several CNN architectures to classify the morphologies of galaxies in both a **3-class** (elliptical, lenticular, spiral) and **4-class** (+irregular/miscellaneous) schema with a dataset of **14034 visually-classified SDSS images**. We develop a new CNN architecture that outperforms existing models in both 3 and 4-way classification, with overall classification accuracies of 83% and 81% respectively. We also compare the accuracies of 2-way / binary classifications between all four classes, showing that **ellipticals and spirals are most easily distinguished** (>98% accuracy), while **spirals and irregulars are hardest to differentiate** (78% accuracy). 
	- Through an analysis of all classified samples, we find tentative evidence that **misclassifications are physically meaningful**, with lenticulars misclassified as ellipticals tending to be more massive, among other trends. 
	- We further combine our binary CNN classifiers to perform a hierarchical classification of samples, obtaining comparable accuracies ($81\%$) to the direct 3-class CNN, but considerably worse accuracies in the 4-way case ($65\%$). As an additional verification, we apply our networks to a **small sample of Galaxy Zoo images**, obtaining accuracies of 92%, 82% and 77% for the binary, 3-way and 4-way classifications respectively.
	- Not bad at all

---



