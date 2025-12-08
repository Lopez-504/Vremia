
# Thesis: Structure formation and such

- <mark>Suto-2015</mark>: CONFRONTATION OF TOP-HAT SPHERICAL COLLAPSE AGAINST DARK HALOS FROM COSMOLOGICAL N-BODY SIMULATIONS

	- The top-hat spherical collapse model (TSC) is one of the most fundamental analytical frameworks to describe the **non-linear growth of cosmic structure**. TSC has motivated, and been widely applied in, various researches even in the current era of precision cosmology. While numerous studies exist to examine its validity against numerical simulations in a statistical fashion, there are few analyses to compare the TSC dynamics in an **individual object-wise basis**, which is what we attempt in the present paper. 
	- We extract $100$ halos at $z = 0$ from a cosmological N-body simulation according to the conventional TSC criterion for the spherical over-density. Then we trace back their spherical counter-parts at earlier epochs. Just prior to the turn-around epoch of the halos, **their dynamics is well approximated by TSC**, but their **turn-around epochs are systematically delayed** and the virial radii are larger by $\sim 20\%$ on average relative to the TSC predictions. 
	- We ﬁnd that this systematic deviation is mainly ascribed to the non-uniformity/inhomogeneity of dark matter density proﬁles and the non-zero velocity dispersions, **both of which are neglected in TSC**. In particular, the **inside-out-collapse** and **shell-crossing** of dark matter halos play an important role in generating the signiﬁcant velocity dispersion. The implications of the present result are brieﬂy discussed. 
	- Subject headings: cosmology; spherical collapse model; dark matter halo

---

- <mark>Herrera 2019</mark>: Top-Hat Spherical Collapse with Clustering Dark Energy. I. Radius Evolution and Critical Contrast Density

	- Understanding the influence of dark energy on the formation of structures is currently a major challenge in Cosmology, since it can distinguish otherwise degenerated viable models. In this work we consider the Top-Hat Spherical-Collapse (SC) model with **dark energy**, which can partially (or totally) cluster, according to a free parameter $\gamma$. 
	- The lack of energy conservation has to be taken into account accordingly, as we will show. We determine characteristic quantities for the SC model, such as the **critical contrast density** and **radius evolution**, with particular emphasis on their dependence on the clustering parameter $\gamma$.

---

- <mark>East 2018</mark>: Comparing fully general relativistic and Newtonian calculations of structure formation

	- In the standard approach to studying cosmological structure formation, the overall expansion of the Universe is assumed to be homogeneous, with the **gravitational effect of inhomogeneities** encoded entirely in a **Newtonian potential**. 
	- A topic of ongoing debate is to what degree this fully captures the dynamics dictated by GR, especially in the era of precision cosmology. To quantitatively assess this, we directly compare standard N-body Newtonian calculations to full numerical solutions of the Einstein equations, for cold matter with various magnitude initial inhomogeneities on scales comparable to the **Hubble horizon**. 
	- We analyze the differences in the evolution of density, luminosity distance, and other quantities defined with respect to **fiducial observers**. This is carried out by reconstructing the effective spacetime and matter fields dictated by the Newtonian quantities, and by taking care to distinguish effects of numerical resolution. 
	- We find that the fully general relativistic and Newtonian calculations show **excellent agreement**, even **well into the nonlinear regime**. They only notably differ in regions where the weak gravity assumption breaks down, which arise when considering extreme cases with perturbations exceeding standard values.

---

- <mark>East 2012</mark>: Hydrodynamics in full general relativity with conservative AMR

	- There is great interest in numerical relativity simulations involving matter due to the likelihood that binary compact objects involving neutron stars will be detected by gravitational wave observatories in the coming years, as well as to the possibility that binary compact object mergers could explain short-duration gamma-ray bursts.
	- We present a **code designed for simulations of hydrodynamics coupled to the Einstein field equations** targeted toward such applications. This code has recently been used to study eccentric mergers of black hole-neutron star binaries. We evolve the fluid conservatively using **high-resolution shock-capturing methods**, while the field equations are solved in the **generalized-harmonic formulation** with finite differences. 
	- In order to resolve the various scales that may arise, we use adaptive mesh refinement (AMR) with grid hierarchies based on truncation error estimates. A noteworthy feature of this code is the implementation of the flux correction algorithm of Berger and Colella to ensure that the conservative nature of fluid advection is respected across AMR boundaries. We present various tests to compare the performance of different limiters and flux calculation methods, as well as to demonstrate the utility of AMR flux corrections.

---

- <mark>Munoz 2022</mark>: EBWeyl: a Code to Invariantly Characterize Numerical Spacetimes

	- Relativistic cosmology can be formulated covariantly, but in dealing with numerical relativity simulations a **gauge choice** is necessary. Although observables should be gauge-invariant, simulations do not necessarily focus on their computations, while it is useful to extract results invariantly. To this end, in order to invariantly characterize spacetimes resulting from cosmological simulations, we present two different methodologies to compute the electric and magnetic parts of the Weyl tensor, $E_{\alpha\beta}$ and $B_{\alpha\beta}$ , from which we construct **scalar invariants** and the **Weyl scalars**. 
	- The first method is geometrical, computing these tensors in full from the metric, and the second uses the 3 + 1 slicing formulation. We developed a code for each method and tested them on five analytic metrics, for which we derived $E_{\alpha\beta}$ and $B_{\alpha\beta}$ and the various scalars constructed from them with computer algebra software. 
	- We find **excellent agreement between the analytic and numerical results**. The **slicing code outperforms the geometrical code** for computational convenience and accuracy; on this basis we make it publicly available in [GitHub](https://github.com/robynlm/ebweyl) with the name EBWeyl. We emphasize that this post-processing code is applicable to any numerical spacetime in any gauge.

---

- <mark>Munoz 2023</mark>: Structure formation and quasi-spherical collapse from initial curvature perturbations with numerical relativity simulations

	- We use numerical relativity simulations to describe the spacetime evolution during nonlinear structure formation in ΛCDM cosmology. Fully nonlinear initial conditions are set at an initial redshift $z\approx300$, based directly on the gauge invariant comoving curvature perturbation $R_c$ commonly used to model early-universe fluctuations. 
	- Assigning a simple 3-D sinusoidal structure to $R_c$ , we then have a lattice of quasi-spherical over-densities **representing idealised dark matter halos connected through filaments and surrounded by voids**. This structure is implemented in the synchronous-comoving gauge, using a pressureless perfect fluid (dust) description of CDM, and then it is fully evolved with the Einstein Toolkit code. 
	- With this, we look into whether the **Top-Hat spherical and homogeneous collapse model** provides a good description of the collapse of over-densities. We find that the Top-Hat is an excellent approximation for the evolution of peaks, where we observe that the shear is negligible and collapse takes place when the linear density contrast reaches the predicted critical value $\delta_C= 1.69$. Additionally, we characterise the outward expansion of the turn-around boundary and show how it depends on the initial distribution of matter, finding that it is faster in denser directions, incorporating more and more matter in the infalling region. 
	- Using the EBWeyl code we look at the distribution of the **electric and magnetic parts of the Weyl tensor**, finding that they're stronger along and around the filaments, respectively. We introduce a method to **dynamically classify the different regions of the simulation box in Petrov types**. With this, we find that the spacetime is of Petrov type I everywhere, as expected, but we can identify the leading order type in each region and at different times. Along the filaments, the leading Petrov type is D, while the centre of the over-densities remains **conformally flat**, type O, in line with the Top-Hat model. 
	- The surrounding region demonstrates a sort of peeling-off in action, with the spacetime transitioning between different Petrov types as non-linearity grows, with production of gravitational waves.
	- Repo: [GitHub](https://github.com/robynlm/ICPertFLRW)

---

- <mark>Bruni 2014</mark>: Non-Gaussian initial conditions in ΛCDM: Newtonian, relativistic, and primordial contributions

	- The goal of the present paper is to set initial conditions for structure formation at non-linear order, consistent with GR, while also allowing for primordial non-Gaussianity. We use the non-linear continuity and Raychaudhuri equations, which together with the non-linear energy constraint **determine the evolution of the matter density fluctuation in GR**. We solve this equations at 1st and 2nd order in a perturbative expansion, recovering and extending previous results derived in the matter-dominated limit and in the Newtonian regime. 
	- We present a second-order solution for the comoving density contrast in a ΛCDM universe, identifying **non-linear contributions coming from the Newtonian growing mode**, **primordial non-Gaussianity** and **intrinsic non-Gaussianity**, due to the essential non-linearity of the relativistic constraint equations. 
	- We discuss the application of these results to initial conditions in N-body simulations, showing that relativistic corrections mimic a non-zero non-linear parameter $f_{NL}$

---

# Numerical Relativity

- <mark>Baumgarte 2012</mark>: Numerical Relativity in Spherical Polar Coordinates: Evolution Calculations with the BSSN Formulation

	- In the absence of symmetry assumptions most NR simulations adopt Cartesian coordinates. While Cartesian coordinates have some desirable properties, **spherical polar coordinates** appear better suited for certain applications, including gravitational collapse and supernova simulations. Development of NR codes in spherical polar coordinates has been hampered by the need to handle the coordinate singularities at the origin and on the axis, for example by **careful regularization of the appropriate variables**. 
	- Assuming **spherical symmetry** and adopting a **covariant version of the BSSN equations**, Montero and Cordero-Carrión recently demonstrated that such a regularization is not necessary when a partially implicit Runge-Kutta (PIRK) method is used for the time evolution of the gravitational fields. Here we report on an **implementation of the BSSN equations in spherical polar coordinates without any symmetry assumptions**. 
	- Using a **PIRK method** we obtain stable simulations in three spatial dimensions without the need to regularize the origin or the axis. We perform and discuss a number of tests to assess the stability, accuracy and convergence of the code, namely **weak gravitational waves**, “hydro-without-hydro” evolutions of spherical and rotating relativistic stars in equilibrium, and single black holes.

---

- <mark>Ruchlin 2018</mark>: SENR/NRPy+: Numerical relativity in singular curvilinear coordinate systems

	- We report on a new open-source, user-friendly NR code package called SENR/NRPy+. Our code extends previous implementations of the BSSN reference-metric for- mulation to a much broader class of curvilinear coordinate systems, making it ideally suited to modeling physical configurations with approximate or exact symmetries. In the context of modeling black hole dynamics, it is orders of magnitude more efficient than other widely used open-source nu- merical relativity codes. NRPy+ provides a Python-based interface in which equations are written in natural tensorial form and output at arbitrary finite difference order as highly efficient C code, putting complex tensorial equations at the scientist’s fingertips without the need for an expensive software license. SENR provides the algorithmic framework that combines the C codes generated by NRPy+ into a functioning numerical relativity code. We validate against two other established, state-of-the-art codes, and achieve excellent agreement. For the first time—in the context of moving puncture black hole evolutions—we demonstrate nearly exponential convergence of constraint violation and gravitational waveform errors to zero as the order of spatial finite difference derivatives is increased, while fixing the numerical grids at moderate resolution in a singular coordinate system. Such behavior outside the horizons is remarkable, as numerical errors do not converge to zero near punctures, and all points along the polar axis are coordinate singularities. The formulation addresses such coordinate singularities via cell-centered grids and a simple change of basis that analytically regularizes tensor components with respect to the coordinates. Future plans include extending this formulation to allow dynamical coordinate grids and bispherical-like distribution of points to efficiently capture orbiting compact binary dynamics

---

# Ray Tracing

- <mark>Cardenas 2024</mark>: Adaptive Analytical Ray Tracing of Black Hole Photon Rings

	- Recent interferometric observations by the Event Horizon Telescope have resolved the horizon-scale emission from sources in the vicinity of nearby supermassive black holes. Future space-based interferometers promise to measure the “photon ring”—a narrow, ring-shaped, lensed feature predicted by general relativity, but not yet observed—and thereby open a new window into strong gravity. 
	- Here we present AART: an Adaptive Analytical Ray-Tracing code that exploits the **integrability of light propagation in the Kerr spacetime** to **rapidly compute high-resolution simulated black hole images**, together with the corresponding radio visibility accessible on very long space-ground baselines. The code samples images on a nonuniform adaptive grid that is specially tailored to the lensing behavior of the Kerr geometry and is therefore particularly well-suited to studying photon rings. 
	- This numerical approach guarantees that interferometric signatures are correctly computed on long baselines, and the modularity of the code allows for **detailed studies of equatorial sources with complex emission profiles and time variability**. 
	- To demonstrate its capabilities, we use AART to simulate a black hole movie of a stochastic, non-stationary, non-axisymmetric equatorial source; by time-averaging the visibility amplitude of each snapshot, we are able to extract the projected diameter of the photon ring and recover the shape predicted by general relativity.

---



# Machine Learning

- <mark>Walmsley 2021</mark>: Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314 000 galaxies

	- Galaxy Zoo DECaLS: detailed visual morphological classifications for Dark Energy Camera Le gac y Surv e y images of galaxies within the SDSS DR8 footprint. Deeper DECaLS images ( r = 23.6 versus r = 22.2 from SDSS) reveal spiral arms, weak bars, and tidal features not previously visible in SDSS imaging. To best exploit the greater depth of DECaLS images, volunteers select from a new set of answers designed to impro v e our sensitivity to mergers and bars. Galaxy Zoo volunteers provide 7.5 million individual classifications o v er 314 000 galaxies. 140 000 galaxies receive at least 30 classifications, sufficient to accurately measure detailed morphology like bars, and the remainder receive approximately 5. All classifications are used to train an ensemble of Bayesian convolutional neural networks (a state-of-the-art deep learning method) to predict posteriors for the detailed morphology of all 314 000 galaxies. We use active learning to focus our volunteer effort on the galaxies which, if labelled, would be most informative for training our ensemble. When measured against confident volunteer classifications, the trained networks are approximately 99 per cent accurate on every question. Morphology is a fundamental feature of every galaxy; our human and machine classifications are an accurate and detailed resource for understanding how galaxies evolve.

---

- <mark>Domı́nguez 2018</mark>: Improving galaxy morphologies for SDSS with Deep Learning

	- We present a morphological catalogue for ∼670,000 galaxies in the SDSS in two flavours: **T-Type**, related to the Hubble sequence, and **Galaxy Zoo 2** (GZ2) classification scheme. By combining accurate existing visual classification catalogues with machine learning, we provide the **largest and most accurate morphological catalogue up to date**. 
	- The classifications are obtained with Deep Learning algorithms using Convolutional Neural Networks (CNNs). We use two visual classification catalogues, GZ2 and Nair & Abraham (2010), for training CNNs with colour images in order to obtain T-Types and a series of **GZ2 type questions** (disk/features, edge-on galaxies, bar signature, bulge prominence, roundness and mergers). 
	- We also provide an additional probability enabling a separation between pure elliptical (E) from S0, where the T-Type model is not so efficient. For the T-Type, our results show smaller offset and scatter than previous models trained with support vector machines. For the GZ2 type questions, our models have large accuracy ($> 97\%$), precision and recall values ($> 90\%$) when applied to a test sample with the same characteristics as the one used for training. The catalogue is publicly released with the paper.

---

- <mark>Barchi 2019</mark>: Machine and Deep Learning applied to galaxy morphology - A comparative study

	- Morphological classification is a key piece of information to define samples of galaxies aiming to study the large-scale structure of the universe. In essence, the challenge is to build up a **robust methodology to perform a reliable morphological estimate from galaxy images**. 
	- Here, we investigate how to substantially improve the galaxy classification within large datasets by **mimicking human classification**. We combine accurate visual classifications from the **Galaxy Zoo project** with machine and deep learning methodologies. 
	- We propose two distinct approaches for galaxy morphology: one based on **non-parametric morphology and traditional ML algorithms**; and another based on **Deep Learning**. 
	- To measure the input features for the traditional machine learning methodology, we have developed a system called **CyMorph**, with a novel non-parametric approach to study galaxy morphology. The main datasets employed comes from the SDSS-DR7. We also discuss the **class imbalance problem** considering three classes. 
	- Performance of each model is mainly measured by Overall Accuracy (OA). A spectroscopic validation with astrophysical parameters is also provided for Decision Tree models to assess the quality of our morphological classification. In all of our samples, both Deep and Traditional Machine Learning approaches have over $94.5\%$ OA to classify galaxies in two classes (elliptical and spiral). We compare our classification with state-of-the-art morphological classification from literature. Considering only two classes separation, we achieve $99\%$ of overall accuracy in average when using our deep learning models, and $82\%$ when using three classes. 
	- We provide a catalog with $670,560$ galaxies containing our best results, including morphological metrics and classification.

---

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

- <mark>Lintott 2010</mark>: Galaxy Zoo 1: Data Release of Morphological Classifications for nearly 900,000 galaxies

	- Morphology is a powerful indicator of a **galaxy’s dynamical and merger history**. It is strongly correlated with many physical parameters, including mass, star formation history and the distribution of mass.
	- The Galaxy Zoo project collected simple morphological classifications of nearly 900,000 galaxies drawn from the SDSS, contributed by hundreds of thousands of volunteers. This large number of classifications allows us to exclude classifier error, and measure the influence of subtle biases inherent in morphological classification. 
	- This paper presents the data collected by the project, alongside measures of classification accuracy and bias. The data are now publicly available and full catalogues can be downloaded from [GalaxyZoo](http://data.galaxyzoo.org)

---

- <mark>Vaswani 2017</mark>: Attention is all you need

	- The dominant sequence transduction models are based on complex recurrent or CNN that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data

---

- <mark>Daoud 2018</mark>: A survey of neural network-based cancer prediction models from microarray data

	- Neural networks are powerful tools used widely for building cancer prediction models from microarray data. We review the most recently proposed models to highlight the roles of neural networks in predicting cancer from gene expression data. We identified articles published between 2013–2018 in scientific databases using key- words such as cancer classification, cancer analysis, cancer prediction, cancer clustering and microarray data. Analyzing the studies reveals that neural network methods have been either used for filtering (data engineering) the gene expressions in a prior step to prediction; predicting the existence of cancer, cancer type or the survivability risk; or for clustering unlabeled samples. This paper also discusses some practical issues that can be considered when building a neural network-based cancer prediction model. Results indicate that the functionality of the neural network determines its general architecture. However, the decision on the number of hidden layers, neurons, hypermeters and learning algorithm is made using trail-and-error techniques.

---



# Cosmology

- <mark>Malik 2009</mark>: Cosmological perturbations

	- We review the study of inhomogeneous perturbations about a homogeneous and isotropic background cosmology. We adopt a coordinate based approach, but give geometrical interpretations of metric perturbations in terms of the **expansion**, **shear** and **curvature of constant-time hypersurfaces** and the **orthogonal timelike vector field**. We give the gauge transformation rules for metric and matter variables at 1st and 2nd order. 
	- We show how gauge invariant variables are constructed by identifying geometric or matter variables in physically-defined coordinate systems, and give the **relations between many commonly used gauge-invariant variables**. In particular we show how the Einstein equations or energy-momentum conservation can be used to obtain simple evolution equations at linear order, and discuss extensions to non-linear order. 
	- We present evolution equations for systems with multiple interacting fluids and scalar fields, identifying **adiabatic and entropy perturbations**. As an application we consider the origin of primordial curvature and isocurvature perturbations from field perturbations during inflation in the very early universe.

---

- <mark>Son 2025</mark>: Strong Progenitor Age-bias in Supernova Cosmology. II. Alignment with DESI BAO and Signs of a Non-Accelerating Universe

	- Supernova (SN) cosmology is based on the key assumption that the luminosity standardization process of Type Ia SNe remains invariant with progenitor age. However, direct and extensive age measurements of SN host galaxies reveal a significant ($5.5\sigma$) correlation between standardized SN magnitude and progenitor age, which is expected to introduce a serious systematic bias with redshift in SN cosmology. 
	- This systematic bias is largely uncorrected by the commonly used mass-step correction, as progenitor age and host galaxy mass evolve very differently with redshift. After correcting for this age-bias as a function of redshift, the SN dataset aligns more closely with the $w_0 w_𝑎$ CDM model recently suggested by the DESI BAO project from a combined analysis using only BAO and CMB data. 
	- This result is further supported by an evolution-free test that uses only SNe from young, coeval host galaxies across the full redshift range. When the three cosmological probes (SNe, BAO, CMB) are combined, we find a significantly stronger ($> 9\sigma$) tension with the ΛCDM model than that reported in the DESI papers, **suggesting a time-varying dark energy equation of state in a currently non-accelerating universe**.

---

- <mark>Hwang 2006</mark>: Second-order perturbations of a zero-pressure cosmological medium: comoving vs. synchronous gauge

	- Except for the presence of gravitational wave source term, the relativistic perturbation equations of a zero-pressure irrotational fluid in a flat Friedmann world model **coincide exactly with the Newtonian ones to the 2nd order in perturbations**. Such a relativistic-Newtonian correspondence is available in a **special gauge condition** (the comoving gauge) in which all the variables are equivalently gauge invariant. 
	- In this work we compare our results with the ones in the synchronous gauge which has been used often in the literature. Although the final equations look simpler in the synchronous gauge, the variables have **remnant gauge modes**. Except for the presence of the gauge mode for the perturbed order variables, however, the equations in the synchronous gauge are gauge invariant and can be exactly identified as the Newtonian hydrodynamic equations in the Lagrangian frame. In this regard, the relativistic equations to the 2nd order in the comoving gauge are the same as the Newtonian hydrodynamic equations in the Eulerian frame. 
	- We resolve several issues related to the two gauge conditions often to fully nonlinear orders in perturbations.

---

- <mark>Silveravalle 2025</mark> (NaP^[1]): Cosmology with a Non-minimally Coupled Dark Matter Fluid I. Background Evolution

	- We explore a cosmological model in which dark matter is non-minimally coupled to gravity at the fluid level. 
	- While typically subdominant compared to Standard Model forces, such couplings may dominate dark matter dynamics. We show that this interaction **modifies the early-time Friedmann equations**, driving a phase of accelerated expansion that can **resolve the horizon and flatness problems without introducing additional fields**. 
	- At even earlier times, the coupling to spatial curvature may give rise to a **cosmological bounce**, replacing the initial singularity of standard cosmology. These results suggest that non-minimally coupled dark matter could offer a **unified framework for addressing both the singularity** and **fine-tuning problems**.

---

- <mark>Haghani 2024</mark>: The first variation of the matter energy-momentum tensor with respect to the metric, and its implications on modified gravity theories

	- The **1st order variation of the matter energy-momentum tensor** $T_{\mu\nu}$ with respect to the metric tensor $g_{\alpha\beta}$ plays an important role in **modified gravity** theories with geometry-matter coupling, and in particular in the $f(R,T)$ modified gravity theory. 
	- We obtain the expression of the variation $\delta T_{\mu\nu} /\delta g_{\alpha\beta}$ for the baryonic matter described by an equation given in a parametric form, with the basic thermodynamic variables represented by the particle number density, and by the specific entropy, respectively. 
	- The first variation of the matter energy-momentum tensor turns out to be **independent on the matter Lagrangian**, and can be expressed in terms of the pressure, the energy-momentum tensor itself, and the matter fluid four-velocity. We apply the obtained results for the case of the $f(R,T)$ gravity theory, where $R$ is the Ricci scalar, and $T$ is the trace of the matter energy-momentum tensor, which thus becomes a **unique theory**, also independent on the choice of the matter Lagrangian. 
	- A simple cosmological model, in which the **Hilbert-Einstein Lagrangian** is generalized through the addition of a term proportional to $T^n$ is considered in detail, and it is shown that it gives a **very good description of the observational values** of the Hubble parameter up to a redshift of $z \approx 2.5$

---

- <mark>Bernardo 2023</mark>: Dark energy by natural evolution: Constraining dark energy using Approximate Bayesian Computation

	- We look at dark energy from a **biology inspired viewpoint** by means of the Approximate Bayesian Computation (ABC) and late time cosmological observations. 
	- We find that **dynamical dark energy** comes out on top, or in the ABC language **naturally selected**, over the standard ΛCDM cosmological scenario. We confirm this conclusion is robust to whether baryon acoustic oscillations and Hubble constant priors are considered. 
	- Our results show that the algorithm prefers low values of the Hubble constant, consistent or at least a few standard deviation away from the CMB estimate, regardless of the priors taken initially in each model. 
	- This supports the result of the traditional MCMC analysis and could be viewed as strengthening evidence for **dynamical dark energy being a more favorable model of late time cosmology**.

---

- <mark>Rubin 2013</mark> (PFS^[2]): The virialization density of peaks with general density profiles under spherical collapse

	- We calculate the **non-linear virialization density** $\Delta_c$ of halos under spherical collapse from peaks with an arbitrary initial and final density profile. This is in contrast to the standard calculation of $\Delta_c$ which assumes **top-hat profiles**. Given our formalism, the non-linear halo density can be calculated once the shape of the initial peak’s density profile and the shape of the virialized halo’s profile are provided. 
	- We solve for $\Delta_c$ for halos in an **Einstein de-Sitter** and **ΛCDM universe**. As examples, we consider power-law initial profiles as well as spherically averaged peak profiles calculated from the statistics of a Gaussian random field. 
	- We find that, depending on the profiles used, $\Delta_c$ is smaller by a factor of a few to as much as a factor of 10 as compared to the density given by the standard calculation ($\approx200$). Using our results, we show that, for halo finding algorithms that identify halos through an over-density threshold, the halo mass function measured from cosmological simulations can be enhanced at all halo masses by a factor of a few.
	- This difference could be important when using numerical simulations to assess the validity of analytic models of the halo mass function.

---

- <mark>Suto 2015</mark>: CONFRONTATION OF TOP-HAT SPHERICAL COLLAPSE AGAINST DARK HALOS FROM COSMOLOGICAL N-BODY SIMULATIONS

	- The top-hat spherical collapse model (TSC) is one of the most fundamental analytical frameworks to describe the **non-linear growth of cosmic structure**. TSC has motivated, and been widely applied in, various researches even in the current era of precision cosmology. While numerous studies exist to examine its validity against numerical simulations in a statistical fashion, there are few analyses to compare the TSC dynamics in an **individual object-wise basis**, which is what we attempt in the present paper. 
	- We extract $100$ halos at $z = 0$ from a cosmological N-body simulation according to the conventional TSC criterion for the spherical over-density. Then we trace back their spherical counter-parts at earlier epochs. Just prior to the turn-around epoch of the halos, **their dynamics is well approximated by TSC**, but their **turn-around epochs are systematically delayed** and the virial radii are larger by $\sim 20\%$ on average relative to the TSC predictions. 
	- We ﬁnd that this systematic deviation is mainly ascribed to the non-uniformity/inhomogeneity of dark matter density proﬁles and the non-zero velocity dispersions, **both of which are neglected in TSC**. In particular, the **inside-out-collapse** and **shell-crossing** of dark matter halos play an important role in generating the signiﬁcant velocity dispersion. The implications of the present result are brieﬂy discussed. 
	- Subject headings: cosmology; spherical collapse model; dark matter halo

---

- <mark>Grandón 2018</mark>: Exploring evidence of interaction between dark energy and dark matter

	- We use the latest observations on SNIa, H(z), BAO, $f_\text{gas}$ in clusters and CMB, to **constrain three models** showing an **explicit interaction between dark matter and dark energy**. 
	- In particular, we use the BOSS BAO measurements at $z ≃ 0.32, 0.57$ and $2.34$, using the full 2-dimensional constraints on the angular and line of sight BAO scale. We find that using all five observational probes together, **two of the interaction models show positive evidence** at more than $3 σ$. 
	- Although significant, further study is needed to establish this statement firmly. (chickens out at the last minute?)

---

- <mark>Giblin 2018</mark>: The Limited Accuracy of Linearized Gravity

	- Standard cosmological models rely on an approximate treatment of gravity, utilizing solutions of the linearized Einstein equations as well as physical approximations. In an era of precision cosmology, we should ask: are these approximate predictions sufficiently accurate for comparison to observations, and can we draw meaningful conclusions about properties of our Universe from them? In this work we examine the accuracy of linearized gravity in the presence of collisionless matter and a cosmological constant utilizing fully general relativistic simulations. We observe the gauge-dependence of corrections to linear theory, and note the amplitude of these corrections. For perturbations whose amplitudes are in line with expectations from the standard ΛCDM model, we find that the full, general relativistic metric is well-described by linear theory in Newtonian and harmonic gauges, while the metric in comoving-synchronous gauge is not. For the largest observed structures in our Universe, our results suggest that corrections to linear gravitational theory can reach or surpass the percent-level.

---

- <mark>Tian 2021</mark>: On the question of measuring spatial curvature in an inhomogeneous universe

	- The curvature of a spacetime, either in a topological sense, or averaged over super-horizon-sized patches, is often equated with the global curvature term that appears in Friedmann’s equation. In general, however, the Universe is inhomogeneous, and gravity is a nonlinear theory, thus any curvature perturbations violate the assumptions of the FLRW model; it is not necessarily true that local curvature, averaged over patches of constant-time surfaces, will reproduce the observational effects of global symmetry. Further, the curvature of a constant-time hypersurface is not an observable quantity, and can only be inferred indirectly. Here, we examine the behavior of curvature modes on hypersurfaces of an inhomogeneous spacetime non-perturbatively in a numerical relativistic setting, and how this curvature corresponds with that inferred by observers. We also note the point at which observations become sensitive to the impact of curvature sourced by inhomogeneities on inferred average properties, finding general agreement with past literature.

---

- <mark>Serrano 2023</mark>: Friedmann equations and cosmic bounce in a modified cosmological scenario

	- Derivation of modified Raychaudhuri and Friedmann equations from a phenomenological model of **quantum gravity** based on the **thermodynamics of spacetime**. Starting from general gravitational equations of motion which encode low-energy quantum gravity effects, we found its particular solution for homogenous and isotropic universes with standard matter content, obtaining a modified Raychaudhuri equation. 
	- Then, we imposed local energy conservation and used a perturbative treatment to derive a modified Friedmann equation. The modified evolution in the early universe we obtained suggests a replacement of the Big Bang singularity by a regular bounce. 
	- Lastly, we also briefly discuss the range of validity of the perturbative approach and its results.

---

- <mark>Farsi 2023</mark>: Evolution of Spherical Overdensities in Energy-Momentum-Squared Gravity

	- Employing the spherical collapse (SC) formalism, we investigate the **linear evolution of the matter overdensity** for **energy-momentum-squared gravity** (EMSG), which in practical phenomenological terms, one may imagine as an extension of the ΛCDM model of cosmology. The underlying model, while still having a cosmological constant, is a non-linear material extension of GR and includes correction terms that are dominant in the high-energy regime, the early universe.
	- Considering the FRW background in the presence of a cosmological constant, we find the effects of the modifications arising from EMSG on the growth of perturbations at the early stages of the universe. Considering both possible negative and positive values of the **model parameter of EMSG**, we discuss its role in the evolution of the matter density contrast and growth function in the level of linear perturbations. 
	- While EMSG leaves imprints distinguishable from ΛCDM, we find that the negative range of the ESMG model parameter is **not well-behaved**, indicating an anomaly in the parameter space of the model. In this regard, for the evaluation of the galaxy cluster number count in the framework of EMSG, we equivalently provide an analysis of the number count of the gravitationally collapsed objects (or the **dark matter halos**). 
	- We show that the galaxy cluster number count decreases compared to the ΛCDM model. In agreement with the hierarchical model of structure formation, in EMSG cosmology the **more massive structures are less abundant**, meaning that form at later times.

---

- <mark>Lucie-Smith 2018</mark>: Machine learning cosmological structure formation

	- We train a ML algorithm to learn cosmological structure formation from N-body simulations. The algorithm infers the **relationship between the initial conditions and the final dark matter haloes**, without the need to introduce approximate halo collapse models. 
	- We gain **insights into the physics driving halo formation** by evaluating the predictive performance of the algorithm when provided with different types of information about the local environment around dark matter particles. 
	- **The algorithm learns to predict whether or not dark matter particles will end up in haloes of a given mass range**, based on spherical overdensities. 
	- We show that the resulting predictions match those of spherical collapse approximations such as extended Press-Schechter theory. 
	- Additional information on the shape of the local gravitational potential is **not able to improve halo collapse predictions**; the linear density field contains sufficient information for the algorithm to also reproduce **ellipsoidal collapse** predictions based on the **Sheth-Tormen model**. 
	- We investigate the algorithm’s performance in terms of halo mass and radial position and perform **blind analyses on independent initial conditions realisations** to demonstrate the generality of our results.

---

- <mark>Riess 1998</mark>: Observational Evidence from Supernovae for an Accelerating Universe and a Cosmological Constant

	- We present spectral and photometric observations of 10 type Ia supernovae (SNe Ia) in the redshift range 0.16 ≤ z ≤ 0.62. The luminosity distances of these objects are determined by methods that employ relations between SN Ia luminosity and light curve shape. Combined with previous data from our High-Z Supernova Search Team, this expanded set of 16 high-redshift supernovae and a set of 34 nearby supernovae are used to place constraints on the following cosmological parameters: the Hubble constant ($H_0$), the mass density ($\Omega_M$), the cosmological constant (i.e., the vacuum energy density, $\Omega_\Lambda$), the deceleration parameter ($q_0$), and the dynamical age of the Universe ($t_0$). 
	- The distances of the high-redshift SNe Ia are, on average, 10% to 15% farther than expected in a low mass density ($\Omega_M=0.2$) Universe without a cosmological constant. 
	- Different light curve fitting methods, SN Ia subsamples, and prior constraints unanimously **favor eternally expanding models with positive cosmological constant** (i.e., $\Omega_\Lambda>0$) and a current acceleration of the expansion (i.e., $q_0<0$). With no prior constraint on mass density other than Ω M ≥ 0, the spectroscopically confirmed SNe Ia are statistically consistent with q 0 < 0 at the 2.8σ and 3.9σ confidence levels, and with $\Omega_\Lambda>0$ at the $3.0\sigma$ and $4.0\sigma$ confidence levels, for two different fitting methods respectively. 
	- Fixing a “minimal” mass density, $\Omega_M=0.2$, results in the weakest detection, Ω Λ > 0 at the 3.0σ confidence level from one of the two methods. For a flat-Universe prior ($\Omega_M$ + $\Omega_\Lambda$ = 1), the spectroscopically confirmed SNe Ia require Ω Λ > 0 at 7σ and 9σ formal significance for the two different fitting methods. A Universe closed by ordinary matter (i.e., $\Omega_M=1$) is formally ruled out at the $7\sigma$ to $8\sigma$ confidence level for the two different fitting methods. 
	- We estimate the dynamical age of the Universe to be $14.2 \pm1.5$ Gyr including **systematic uncertainties in the current Cepheid distance scale**. 
	- We **estimate the likely effect of several sources of systematic error**, including: 
		- Progenitor and metallicity evolution
		- Extinction
		- Sample selection bias
		- Local perturbations in the expansion rate
		- Gravitational lensing
		- Sample contamination
		- Presently, none of these effects reconciles the data with $\Omega_\Lambda=0$ and $q_0\geq0$.

---

- <mark>Ávila 2025</mark>: Transverse BAO scale measurement at $z_\text{eff}= 1.725$ with the **SDSS** quasars catalog

	- Studying the **SDSS-DR16** quasar catalog, we detect a baryon acoustic oscillation (BAO) signal in the **two-point angular correlation function** with a [[statistical significance]] of **3σ**, at an effective redshift of $z_\text{eff}= 1.725$. Using a simple parameterization—comprising a polynomial plus a Gaussian function—we measure the transverse BAO scale as $θ_\text{BAO} = 1.928^\circ \pm 0.094^\circ$. This measurement is obtained from a narrow redshift shell, $z \in [1.72, 1.73]$ (i.e., ∆z = 0.01), **thin enough that projection-effect corrections are negligible**, making it only weakly dependent on the assumed [[fiducial]] cosmology. The only assumption adopted is isotropy in the computation of the correlation function, further ensuring that the result depends only weakly on specific cosmological-model hypotheses. 
	- We also investigate possible systematics that could affect the detection or significance of the BAO signal and find them to be subdominant or implausible. 
	- When combined with other transverse BAO measurements from the literature, our result shows **good concordance**—within the **1σ confidence level**—with the cosmological parameter values reported by the **Planck and DESI collaborations**. 
	- This new measurement of the transverse BAO scale, obtained from the SDSS quasar sample with minimal cosmological-model assumptions, provides an additional independent constraint for updated statistical studies aimed at probing the nature of dark energy.

---

- <mark>Villanueva 2015</mark>: The generalized Chaplygin–Jacobi gas

	- The present paper is devoted to find a new generalization of the generalized Chaplygin gas. Therefore, starting from the **Hubble parameter associated to the Chaplygin scalar field** and using some **elliptic identities**, the elliptic generalization is straightforward. 
	- Thus, all relevant quantities that drive inflation are calculated exactly. Finally, using the measurement on inflation from the Planck 2015 results, observational constraints on the parameters are given.

---

- <mark>Carroll 2000</mark>: The Cosmological Constant

	- Review of the physics and cosmology of the cosmological constant. Focusing on recent developments, I present a pedagogical overview of cosmology in the presence of a cosmological constant, observational constraints on its magnitude, and the physics of a small (and potentially nonzero) vacuum energy.

---

- <mark>Langlois 2010</mark>: A geometrical approach to nonlinear perturbations in relativistic cosmology

	- We give a pedagogical review of a covariant and fully non-perturbative approach to study nonlinear perturbations in cosmology. In the first part, devoted to **cosmological fluids**, we define a **nonlinear extension of the uniform-density curvature perturbation** and derive its evolution equation. In the second part, we focus our attention on multiple scalar fields and present a nonlinear description in terms of adiabatic and entropy perturbations. 
	- In both cases, we show how the formalism presented here enables one to easily obtain **equations up to second, third and higher orders**.
---

# Gravity

- <mark>Castillo-Felisola 2024</mark>: A polynomial affine model of gravity: after ten years

	- The polynomial aﬃne model of gravity was proposed as an alternative to metric and metric-aﬃne gravitational models. What at the beginning was thought as a source of unpredictability, the presence of many terms in the action, turned out to be a milestone, since it contains all possible combinations of the ﬁelds compatible with the **covariance under diﬀeomorphisms**. 
	- We present a review of the advances in the analysis of the model after ten years of its proposal, and sketch the guideline of our future perspectives.

---

- <mark>Castillo-Felisola 2023</mark>: Does the metric play a fundamental role in the building of gravitational models?

	- The idea that GR could be an effective model, of a yet unknown theory of gravity, has gained momentum among theoretical physicists. 
	- The **polynomial affine model of gravity** is an alternative model of affine gravity that possesses many desirable features to pursue a **quantum theory of gravitation**. 
	- In this paper we argue that such features are a consequence of the lack of a metric structure in the building of the model, even though a emergent metric could be defined. The model introduces additional degrees of freedom associated to the geometric properties of the space, which might shed light to understand the nature of the dark sector of the Universe. 
	- When the model is coupled to a scalar field, it is possible to define inflationary scenarios.

---

- <mark>Carballo-Rubio 2022</mark>: Unimodular Gravity vs General Relativity: A status report

	- Unimodular Gravity (UG) is an alternative to General Relativity (GR) which, however, is so closely related to the latter that one can wonder to what extent they are different. The different behavior of the cosmological constant in the semiclassical regimes of both frameworks suggests the possible existence of additional contrasting features. 
	- UG and GR are based on two different gauge symmetries: UG is based on **transverse diffeomorphisms** and **Weyl rescalings** (WTDiff transformations), whereas GR is based on the **full group of diffeomorphisms**. This difference is related to the existence of a fiduciary background structure, a fixed volume form, in UG theories. In this work we present an overview as complete as possible of situations and regimes in which one might suspect that some differences between these two theories might arise. This overview contains analyses in the classical, semiclassical, and quantum regimes. When a particular situation is well known we make just a brief description of its status. For situations less analyzed in the literature we provide here more complete analyses. Whereas some of these analyses are sparse through the literature, many of them are new. Apart from the completely different treatment they provide for the cosmological constant problem, our results uncover no further differences between them. 
	- We conclude that, to the extent that the technical naturalness of the cosmological constant is regarded as a fundamental open issue in modern physics, UG is preferred over GR since the cosmological constant is technically natural in the former.

---

- <mark>Barrera 2020</mark>: Vector modes in ΛCDM: the gravitomagnetic potential in dark matter haloes from relativistic N-body simulations

	- We investigate the transverse modes of the gravitational and velocity fields in ΛCDM, based on a high-resolution simulation performed using the AMR general-relativistic N-body code **GRAMSES**. 
	- We study the generation of vorticity in the dark matter velocity field at low redshift, providing fits to the shape and evolution of its **power spectrum over a range of scales**. By analysing the gravitomagnetic vector potential, which is absent in Newtonian simulations, in dark matter haloes with masses ranging from $∼10^{12.5} h^{-1} M_\text{sun}$ to $∼10^{15} h^{−1} M_\text{sun}$, we find that its magnitude correlates with the halo mass, peaking in the inner regions. Nevertheless, on average, its ratio against the scalar gravitational potential remains fairly constant, below percent level, decreasing roughly linearly with redshift and showing a weak dependence on halo mass. 
	- Furthermore, we show that the gravitomagnetic acceleration in haloes peaks towards the core and reaches almost $10^{−10} h \text{ cm}/s^2$ in the most massive halo of the simulation. However, regardless of the halo mass, the ratio between the gravitomagnetic force and the standard gravitational force is typically at around the $10^{−5}$ level inside the haloes, again without significant radius dependence. 
	- This result confirms that the **gravitomagnetic effects have negligible impact on structure formation**, even for the most massive structures, although its behaviour in low density regions remains to be explored. Likewise, the impact on observations remains to be understood in the future.

---

- <mark>Barrera 2020</mark>: GRAMSES: a new route to general relativistic N -body simulations in cosmology. Part I. Methodology and code description

	- We present GRAMSES, a new **pipeline for nonlinear cosmological N-body simulations in GR**. This code adopts the Arnowitt-Deser-Misner (ADM) formalism of GR, with constant mean curvature and **minimum distortion gauge fixings**, which provides a **fully nonlinear and background independent framework for relativistic cosmology**. 
	- Employing a fully constrained formulation, the Einstein equations are reduced to a set of **10 elliptical equations** which are solved using multigrid relaxation with adaptive mesh refinements (AMR), and **3 hyperbolic equations** for the evolution of tensor degrees of freedom. The current version of GRAMSES neglects the latter by using the **conformal flatness approximation**, which allows it to compute the two scalar and two vector degrees of freedom of the metric. 
	- Here we describe the methodology, implementation, code tests and first results for cosmological simulations in a **ΛCDM universe**, while the generation of initial conditions and physical results will be discussed elsewhere. Inheriting the efficient AMR and massive parallelisation infrastructure from the publicly-available N -body and hydrodynamic simulation code RAMSES , GRAMSES is ideal for studying the detailed behaviour of spacetime inside virialised cosmic structures and hence accurately quantifying the impact of backreaction effects on the cosmic expansion, as well as for investigating GR effects on cosmological observables using cosmic-volume simulations.

---

- <mark>Adamek 2020</mark>: Numerical solutions to Einstein’s equations in a shearing-dust Universe: a code comparison

	- A number of codes for general-relativistic simulations of cosmological structure formation have been developed in recent years. Here we demonstrate that a sample of these codes produce **consistent results beyond the Newtonian regime.** 
	- We simulate solutions to Einstein’s equations **dominated by gravitomagnetism** — a vector-type gravitational field that doesn’t exist in Newtonian gravity and produces **frame-dragging**, the **leading-order post-Newtonian effect**. 
	- We calculate the coordinate-invariant effect on intersecting null geodesics by performing **ray tracing** in each independent code. With this observable quantity, we assess and compare each code’s ability to compute relativistic effects.

---

- <mark>Barausse 2020</mark>: Prospects for Fundamental Physics with LISA

	- In this paper, which is of programmatic rather than quantitative nature, we aim to further delineate and sharpen the **future potential of the LISA** (Laser Interferometer Space Antenna) mission in the area of **fundamental physics**. Given the very broad range of topics that might be relevant to LISA, we present here a sample of what we view as particularly promising fundamental physics directions. 
	- We organize these directions through a “science-first” approach that allows us to **classify how LISA data can inform theoretical physics** in a variety of areas. For each of these theoretical physics classes, we identify the sources that are currently expected to provide the principal contribution to our knowledge, and the areas that need further development. 
	- The classification presented here should not be thought of as cast in stone, but rather as a **fluid framework that is amenable to change with the flow of new insights in theoretical physics**.

---

- <mark>Ilkhchi 2025</mark>: Thermodynamics of FLRW universe in Quadratic Gravity

	- Investigate the thermodynamic aspects of quadratic gravity in a D-dimensional FLRW universe. 
	- First, we derive the field equations and the effective energy-momentum tensor for quadratic gravity. Then, using these equations, we obtain the generalized Misner-Sharp energy within the framework of this model. We consider the thermodynamic behavior of the **apparent horizon** and derive the equations of state related to the pressure, temperature, and radius of the apparent horizon. Using the thermodynamic pressure, we obtain the critical points corresponding to **phase transitions**. 
	- We determine the critical temperature and critical radius in terms of model parameters, including the quadratic coupling and the cosmological constant. We also examine key thermodynamic quantities, such as Wald entropy, specific heat at constant pressure, enthalpy, Gibbs free energy. By examining the behavior of these quantities, we can gain insight into the thermodynamic stability of the quadratic gravity model. In particular, we find that quadratic terms change the stability conditions and can lead to new thermodynamic behaviors compared to GR.

---

- <mark>Miller 2009</mark>: The Relation Between Equal-Time and Light-Front Wave Functions

	- The relation between equal-time and light-front wave functions is studied using models for which the four-dimensional solution of the Bethe-Salpeter wave function can be obtained. The popular prescription of defining the longitudinal momentum fraction using the instant-form free kinetic energy and third component of momentum is found to be incorrect except in the non-relativistic limit. The only presently known way to obtain light-front wave functions from rest-frame, instant-form wave functions is to boost the latter wave functions to the infinite momentum frame. Despite this fact, we prove a relation between certain integrals of the equal-time and light-front wave functions.

---




# Simulations

- <mark>Lipowski 2023</mark>: Heat-Bath and Metropolis Dynamics in Ising-like Models on Directed Regular Random Graphs

	- Using a **single-site** **mean-field approximation** (MFA) and Monte Carlo simulations, we examine **Ising-like** models on **directed regular random graphs**. The models are directed-network implementations of the Ising model, Ising model with absorbing states, and majority voter models. 
	- When these nonequilibrium models are driven by the heat-bath dynamics, their stationary characteristics, such as magnetization, are correctly reproduced by MFA as confirmed by Monte Carlo simulations. It turns out that **MFA reproduces the same result as the generating functional analysis** that is expected to provide the exact description of such models. 
	- We argue that on directed regular random graphs, the neighbors of a given vertex are typically uncorrelated, and that is why MFA for models with heat-bath dynamics provides their exact description. 
	- For models with **Metropolis dynamics**, certain additional correlations become relevant, and MFA, which neglects these correlations, is **less accurate**. 
	- Models with heat-bath dynamics undergo **continuous phase transition**, and at the critical point, the power-law time decay of the order parameter exhibits the behavior of the Ising mean-field universality class. 
	- Analogous phase transitions for models with Metropolis dynamics are **discontinuous**

---

- <mark>Yang 2019</mark>: High Performance Monte Carlo Simulation of Ising Model on TPU Clusters

	- Large-scale deep learning benefits from an emerging class of **AI accelerators**. Some of these accelerators’ designs are general enough for compute-intensive applications beyond AI and **Cloud TPU** is one such example. 
	- In this paper, we demonstrate a novel approach using **TensorFlow** on Cloud TPU to simulate the 2D Ising Model. TensorFlow and Cloud TPU framework enable the simple and readable code to express the complicated distributed algorithm without compromising the performance. Our code implementation fits into a small Jupyter Notebook and fully utilizes **Cloud TPU’s efficient matrix operation** and dedicated high speed inter-chip connection. 
	- The performance is highly competitive: it outperforms the best published benchmarks to our knowledge by $60\%$ in single-core and $250\%$ in multi-core with good linear scaling. When compared to Tesla V100 GPU, the single-core performance maintains a $\sim10\%$ gain. 
	- We also demonstrate that using low precision arithmetic—bfloat16—does not compromise the correctness of the simulation results.
	- Google's GitHub Repo: [link](https://github.com/google-research/google-research/tree/master/simulation_research/ising_model)

---

- <mark>Preis 2009</mark>: GPU accelerated Monte Carlo simulation of the 2D and 3D Ising model

	- The compute uniﬁed device architecture (CUDA) is a programming approach for performing scientiﬁc calculations on a GPU as a data-parallel computing device. The programming interface allows to implement algorithms using extensions to standard C language. With continuously increased number of cores in combination with a high memory bandwidth, a recent GPU offers incredible resources for general purpose computing. 
	- First, we apply this new technology to Monte Carlo simulations of the **2D ferromagnetic square lattice Ising model**. By implementing a variant of the **checkerboard algorithm**, results are obtained up to **60 times faster on the GPU than on a current CPU core**. An implementation of the 3D ferromagnetic cubic lattice Ising model on a GPU is able to generate results up to **35 times faster** than on a current CPU core. 
	- As proof of concept we calculate the **critical temperature** of the 2D and 3D Ising model using **ﬁnite-size scaling techniques**. Theoretical results for the 2D Ising model and previous simulation results for the 3D Ising model can be reproduced.

---

- <mark>Biciusca 2015</mark>: Simulation of liquid–vapour phase separation on GPUs using Lattice Boltzmann models with off-lattice velocity sets

	- We use a 2D Lattice Boltzmann model to investigate the liquid–vapour phase separation in an isothermal van der Waals ﬂuid. 
	- The model is based on the expansion of the distribution function up to the third order in terms of **Hermite polynomials**. 
	- In two dimensions, this model is an **off-lattice one** and has 16 velocities. **The Corner Transport Upwind Scheme** is used to evolve the corresponding distribution functions on a square lattice. The resulting code allows one to follow the liquid–vapour phase separation on lattices up to 4096 × 4096 nodes using a Tesla M2090 GPU

---

- <mark>Roth 2023</mark>: Critical dynamics in a real-time formulation of the functional renormalization group

	- We present first calculations of critical spectral functions of the relaxational Models A, B, and C in the Halperin-Hohenberg classification using a **real-time formulation of the functional renormalization group** (FRG). 
	- We revisit the prediction by Son and Stephanov that the linear coupling of a conserved density to the non-conserved order parameter of Model A gives rise to critical Model-B dynamics. We formulate both 1-loop and 2-loop self-consistent expansion schemes in the 1PI vertex functions as truncations of the effective average action suitable for real-time applications, and analyze in detail how the different critical dynamics are properly incorporated in the framework of the FRG on the closed-time path. 
	- We present results for the corresponding critical spectral functions, extract the dynamic critical exponents for Models A, B, and C, in two and three spatial dimensions, respectively, and compare the resulting values with recent results from the literature.

---

- <mark>Samlodia 2024</mark>: Phase diagram of generalized XY model using tensor renormalization group

	- We use the higher-order tensor renormalization group method to study the **2D generalized XY model** that admits **integer and half-integer vortices**. This model is the deformation of the classical XY model and has a rich phase structure consisting of **nematic**, **ferromagnetic**, and disordered phases and three transition lines belonging to the Berezinskii-Kosterlitz-Thouless (BKT) and Ising class. 
	- We explore the model for a wide range of temperatures and the deformation parameter (∆) and **compute specific heat along with integer and half-integer magnetic susceptibility**, finding both **BKT-like and Ising-like transitions** and the region where they meet.

---

- <mark>Lis 2012</mark>: (NaP^[1]) GPU ­Based Massive Parallel Kawasaki Kinetics In Monte Carlo Modelling of Lipid Microdomains

	- This paper introduces novel method of simulation of lipid biomembranes based on **Metropolis­Hastings** algorithm and Graphic Processing Unit computational power. 
	- Method gives up to 55 times computational boost in comparison to classical computations. Extensive study of algorithm correctness is provided. Analysis of simulation results and results obtained with classical simulation methodologies are presented.

---

- <mark>Finkenrath 2022</mark>: Tackling critical slowing down using global correction steps with equivariant flows: the case of the Schwinger model

	- We propose a **new method for simulating lattice gauge theories in the presence of fermions**. The method combines flow-based generative models for local gauge field updates and hierarchical updates of the factorized fermion determinant. The flow-based generative models are restricted to proposing updates to gauge-fields within subdomains, thus **keeping training times moderate** while increasing the global volume. 
	- We apply our method performs to the **2D Schwinger model** with $N_f = 2$ Wilson Dirac fermions and show that **no critical slowing down is observed** in the sampling of topological sectors up to $\beta = 8.45$. 
	- Furthermore, we show that fluctuations can be suppressed exponentially with the distance between active subdomains, allowing us to achieve **acceptance rates of up to** $99\%$ for the outer-most accept/reject step on lattices volumes of up to $V = 128 \times 128$.

---

- <mark>Bauerschmidt 2024</mark>: Kawasaki dynamics beyond the uniqueness threshold

	- Glauber dynamics of the Ising model on a random regular graph is known to mix fast below the tree uniqueness threshold and exponentially slowly above it. We show that **Kawasaki dynamics of the canonical ferromagnetic Ising model** on a random d-regular graph mixes fast beyond the tree uniqueness threshold when $d$ is large enough (and conjecture that it mixes fast up to the tree reconstruction threshold for all $d \geq 3$). This result follows from a more general spectral condition for (modified) **log-Sobolev inequalities for conservative dynamics of Ising models**. The proof of this condition in fact extends to perturbations of distributions with **log-concave generating polynomial**.

---

- <mark>Lee 2024</mark>: Parallelising Glauber dynamics

	- For distributions over discrete product spaces $\Pi_{i=1}^n\Omega_{i}'$, Glauber dynamics is a Markov chain that at each step, resamples a random coordinate conditioned on the other coordinates. 
	- We show that k-Glauber dynamics, which resamples a random subset of k coordinates, mixes k times faster in $\chi^2$-divergence, and assuming approximate tensorization of entropy, mixes k times faster in KL-divergence. We apply this to obtain parallel algorithms in two settings: 
	- (1) For the **Ising model** $$µ_{J,h} (x) \propto \exp\left(\frac{1}{2} \langle x, Jx\rangle + \langle h, x \rangle\right)$$with $||J|| < 1 − c$ (the regime where fast mixing is known), we show that we can implement each step of $\tilde{\Theta}(n/|J|_F)$-Glauber dynamics efficiently with a parallel algorithm, resulting in a parallel algorithm with running time $\tilde{O}(||J||_F) = \tilde{O}(\sqrt{n})$. 
	- (2) For the mixed **p-spin model at high enough temperature**, we show that with high probability we can implement each step of $\tilde{\Theta}(\sqrt{n})$-Glauber dynamics efficiently and obtain running time $\tilde{O}(\sqrt{n})$

---

- <mark>Romero 2019</mark>: A Performance study of the 2D Ising Model on GPU 

	- The simulation of the 2D Ising model is used as a **benchmark to show the computational capabilities of GPUs**. The rich programming environment now available on GPUs and flexible hardware capabilities allowed us to quickly experiment with several implementation ideas: a simple stencil-based algorithm, recasting the stencil operations into matrix multiplies to take advantage of Tensor Cores available on NVIDIA GPUs, and a highly optimized multi-spin coding approach. Using the managed memory API available in CUDA allows for simple and efficient distribution of these implementations across a multi-GPU NVIDIA DGX-2 server. 
	- We show that even a **basic GPU implementation can outperform current results published on TPUs** and that the optimized multi-GPU implementation can simulate very large lattices **faster than custom FPGA solutions**
	- NVIDIA GitHub Repo: [Link](https://github.com/NVIDIA/ising-gpu)

---

- <mark>Samlodia 2024</mark>: Phase diagram of generalized XY model using tensor renormalization group

	- We use the higher-order tensor renormalization group method to study the **2D generalized XY model** that admits integer and half-integer vortices. This model is the deformation of the classical XY model and has a rich phase structure consisting of nematic, ferromagnetic, and disordered phases and three transition lines belonging to the Berezinskii-Kosterlitz-Thouless (BKT) and Ising class. 
	- We explore the model for a wide range of temperatures, and the deformation parameter, and compute specific heat along with integer and half-integer magnetic susceptibility, finding both **BKT-like** and **Ising-like transitions** and the region where they meet.

---

- <mark>Ferreras 2025</mark>: Simulation of the 1d XY model on a quantum computer

	- The field of quantum computing has grown fast in recent years, both in theoretical advancements and the practical construction of quantum computers. These computers were initially proposed, among other reasons, to efficiently simulate and comprehend the complexities of quantum physics. In this paper, we present a comprehensive scheme for the exact simulation of the 1D XY model on a quantum computer. 
	- We successfully diagonalize the proposed Hamiltonian, enabling access to the complete energy spectrum. Furthermore, we **propose a novel approach to design a quantum circuit to perform exact time evolution**. 
	- Among all the possibilities this opens, we compute the ground and excited state energies for the symmetric XY model with spin chains of n=4 and n=8 spins.
	- Further, we calculate the expected value of transverse magnetization for the ground state in the **transverse Ising model**. Both studies allow the observation of a **quantum phase transition from an antiferromagnetic to a paramagnetic state**. 
	- Additionally, we have simulated the time evolution of the state all spins up in the transverse Ising model. 
	- **The scalability and high performance of our algorithm make it an ideal candidate for benchmarking purposes**, while also laying the foundation for simulating other integrable models on quantum computers

---

- <mark>Mathis 2017</mark>: A Thermodynamically consistent model of a liquid-vapor fluid with a gas

	- This work is devoted to the consistent modeling of a three-phase mixture of a gas, a liquid and its vapor. Since the gas and the vapor are miscible, the mixture is subjected to a non-symmetric constraint on the volume. 
	- Adopting the **Gibbs formalism**, the study of the extensive equilibrium entropy of the system allows to **recover the Dalton’s law between the two gaseous phases**. In addition, we distinguish whether phase transition occurs or not between the liquid and its vapor. The thermodynamical equilibria are described both in extensive and intensive variables. In the latter case, we focus on the geometrical properties of equilibrium entropy. 
	- The consistent characterization of the thermodynamics of the three-phase mixture is used to introduce two **Homogeneous Equilibrium Models** (HEM) depending on mass transfer is taking into account or not. Hyperbolicity is investigated while analyzing the entropy structure of the systems. 
	- Finally we propose two **Homogeneous Relaxation Models** (HRM) for the three-phase mixtures with and without phase transition. Supplementary equations on mass, volume and energy fractions are considered with appropriate source terms which model the relaxation towards the thermodynamical equilibrium, in agreement with entropy growth criterion.

---

- <mark>Burgos 2023</mark>: A Deep Potential model for liquid-vapor equilibrium and cavitation rates of water

	- Computational studies of liquid water and its phase transition into vapor have traditionally been performed using classical water models. Here we utilize the Deep Potential methodology —a ML approach— to study this ubiquitous phase transition, starting from the phase diagram in the liquid-vapor coexistence regime. The machine learning model is trained on ab initio energies and forces based on the SCAN density functional which has been previously shown to reproduce solid phases and other properties of water. Here, we compute the surface tension, saturation pressure and enthalpy of vaporization for a range of temperatures spanning from 300 to 600 K, and evaluate the Deep Potential model performance against experimental results and the semi-empirical TIP4P/2005 classical model. Moreover, by employing the seeding technique, we evaluate the free energy barrier and nucleation rate at negative pressures for the isotherm of 296.4 K. We find that the nucleation rates obtained from the Deep Potential model deviate from those computed for the TIP4P/2005 water model, due to an underestimation in the surface tension from the Deep Potential model. 
	- From analysis of the seeding simulations, we also evaluate the **Tolman length** for the Deep Potential water model, which is ($0.091 \pm 0.008$) nm at 296.4 K. Lastly, we identify that water molecules display a preferential orientation in the liquid-vapor interface, in which H atoms tend to point towards the vapor phase to maximize the enthalpic gain of interfacial molecules. We find that this behaviour is more pronounced for planar interfaces than for the curved interfaces in bubbles. 
	- This work represents the first application of Deep Potential models to the study of liquid-vapor coexistence and water cavitation.

---

- <mark>Jha 2023</mark>: GPU-acceleration of tensor renormalization with PyTorch using CUDA

	- We show that numerical computations based on tensor renormalization group (TRG) methods can be signiﬁcantly accelerated with PyTorch on graphics processing units (GPUs) by leveraging NVIDIA’s Compute Uniﬁed Device Architecture (CUDA). 
	- We ﬁnd improvement in the runtime (for a given accuracy) and its scaling with bond dimension for two-dimensional systems. Our results establish that utilization of GPU resources is essential for future precision computations with TRG.

---

- <mark>Jin 2025</mark>: Supercritical fluids as a distinct state of matter characterized by sub-short-range structural order

	- A supercritical fluid (SCF) – the state of matter at temperatures and pressures above the critical point – exhibits properties intermediate between those of a liquid and a gas. However, whether it constitutes a fundamentally distinct phase or merely a continuous extension of the liquid and gas states remains an open question. Here we show that a **SCF is defined by sub-short-range** (SSR) **structural order** in the spatial arrangement of particles, setting it apart from the gas (disordered), liquid (short-range ordered), and solid (long-range ordered) states. 
	- The SSR structural order can be characterized by a length scale effectively quantified by the number of observable peaks in the radial distribution function. This length grows from a minimum microscopic value, on the order of the inter-particle distance at the gas–SCF boundary, to a diverging value at the SCF–liquid boundary. Based on the emergence of SSR order, we demonstrate that the transport and dynamical properties of the SCF state, including the diffusion coefficient, shear viscosity, and velocity autocorrelation function, also clearly distinguish it from both the liquid and gas states. 
	- **Theoretical predictions are validated by molecular dynamics simulations** of argon and further supported by existing experimental evidence. Our study confirms and assigns physical significance to the refined phase diagram of matter in the supercritical region, consisting of **three distinct states** (gas, supercritical fluid, and liquid) separated by two crossover boundaries that follow universal scaling laws.

---

- <mark>Zarei 2019</mark>: Ising order parameter and topological phase transitions: Toric code in a uniform magnetic field

	- Quantum Ising model in a transverse field is of the simplest quantum many-body systems used for studying universal properties of quantum phase transitions. Interestingly, it is well-known that such phase transitions can be mapped to topological phase transitions in Toric code models. Therefore, one can expect that well-known properties of the **transverse Ising model are used for characterizing topological phase transition in Toric code model**. 
	- In this paper, we consider the magnetization of Ising model and show while it is a **local order parameter**, it is **mapped to a non-local order parameter in the topological side**. Consequently, we introduce a string order parameter for topological phase transitions in Toric code models defined on different lattices with different dimensions in presence of a uniform magnetic field. Since such string order parameter is dual of the Ising order parameter with well-known properties, it leads to a topological (non-local) characterization of phase transition in the Toric code model in uniform field. Our results show that although topological phase transitions do not follow a symmetry-breaking mechanism, it might be still possible to use concepts of symmetry- breaking phase transitions for topological ones by finding suitable mappings.

---

- <mark>Binder 2001</mark>: Monte Carlo tests of renormalization-group predictions for critical phenomena in Ising models

	- A critical review is given of status and perspectives of Monte Carlo simulations that address **bulk and interfacial phase transitions of ferromagnetic Ising models**. 
	- First, some basic methodological aspects of these simulations are briefly summarized (single-spin flip vs. cluster algorithms, finite-size scaling concepts), and then the application of these techniques to the nearest-neighbor Ising model in $d=3$ and $5$ dimensions is described, and a detailed comparison to theoretical predictions is made. 
	- In addition, the case of Ising models with a large but finite range of interaction and the crossover scaling from mean-field behavior to the Ising universality class are treated. If one considers instead a **long-range interaction** described by a power-law decay, new classes of critical behavior depending on the exponent of this power law become accessible, and a stringent test of the $\epsilon$-expansion becomes possible. 
	- As a final type of crossover from mean-field type behavior to 2D Ising behavior, the interface localization-delocalization transition of Ising films confined between "competing" walls is considered. This problem is still hampered by questions regarding the appropriate coarse-grained model for the fluctuating interface near a wall, which is the starting point for both this problem and the theory of critical wetting

---

- <mark>CHERRINGTON 2008</mark>: A DUAL ALGORITHM FOR NON-ABELIAN YANG-MILLS COUPLED TO DYNAMICAL FERMIONS

	- We extend the dual algorithm recently described for pure, non-abelian Yang-Mills on the lattice to the case of lattice fermions coupled to Yang-Mills, by constructing an ergodic Metropolis algorithm for dynamic fermions that is local, exact, and built from gauge-invariant boson-fermion coupled configurations. Fon concreteness, we present in detail the case of 3D, for the group SU(2) and staggered fermions, however the algorithm readily generalizes with regard to group and dimension. 
	- The treatment of the fermion determinant makes use of a polymer expansion; as with previous proposals making use of the polymer expansion in higher than 2D, the critical question for practical applications is whether the presence of negative amplitudes can be managed in the continuum limit.

---

- <mark>Marugan 2018</mark>: A survey of artiﬁcial neural network in wind energy systems

	- Wind energy has become one of the most important forms of renewable energy. Wind energy conversion systems are more sophisticated and new approaches are required based on advance analytics. 
	- This paper presents an exhaustive **review of ANNs used in wind energy systems**, identifying the methods most employed for diﬀerent applications and demonstrating that Artiﬁcial Neural Networks can be an alternative to conventional methods in many cases. More than 85% of the 190 references employed in this paper have been published in the last 5 years. The methods are classiﬁed and analysed into four groups according to the application: forecasting and predictions; design optimization; fault detection and diagnosis; and optimal control. A statistical analysis of the current state and future trends in this ﬁeld is carried out. An analysis of each application group about the strengths and weaknesses of each ANN structure is carried out. A quantitative analysis of the main references is carried out showing new statistical results of the current state and future trends of the topic. The paper describes the main challenges and technological gaps concerning the application of ANN to wind turbines, according to the literature review. An overall table is provided to summarize the most important references according to the application groups and case studies.



# Weather/Atmospheric Science

- <mark>Gu 2025</mark>: Seeing and Meteorological Analysis at the North-1 and North-2 Points of Muztagh-Ata Site on the Pamir Plateau

	- To support the selection of large optical/infrared telescope sites in western China, **long-term monitoring** of atmospheric conditions and **astronomical seeing** has been conducted at the Muztagh-Ata site on the Pamir Plateau since 2017. With the monitoring focus gradually shifting northward, three stations were established: the South Point, North-1 point, and North-2 point. The North-1 point, selected as the site for the Muztagh-Ata 1.93 m Synergy Telescope (MOST), has recorded seeing and meteorological parameters since late 2018. In 2023, the North-2 point was established approximately 1.5 km northeast of North-1 point as a candidate location for a future large-aperture telescope. A 10m DIMM tower and a PC-4A environmental monitoring system were installed to evaluate site quality. This study presents a comparative analysis of data from the North-1 and North-2 points during 2018–2024. The median seeing is 0.89 ′′ at North-1 and 0.78 ′′ at North-2. Both points show clear seasonal and diurnal variations, with winter nights offering optimal observing conditions. On average, about 64% of the nighttime duration per year is suitable for astronomical observations. Nighttime temperature variation is low: 2.03 ◦ C at North-1 and 2.10 ◦ C at North-2. Median wind speeds are 5–6 m/s, with dominant directions between 210 ◦ and 300 ◦ , contributing to stable airflow. Moderate wind suppresses turbulence, while strong shear and rapid fluctuations degrade image quality. These findings confirm that both the North-1 and North-2 points offer high-quality atmospheric conditions and serve as promising sites for future ground-based optical/infrared telescopes in western China.

---

- <mark>Lazcano 2024</mark>: (Thesis) ASTRONOMICAL SEEING PREDICTION AT PARANAL OBSERVATORY

	- Astronomical seeing is an optical turbulence variable that refers to the distorting effect of the atmosphere on images captured by observatories of astronomical objects. This parameter can be calculated in real-time using different instruments, including the Differential Image Motion Monitor (DIMM). However, observatories usually require a prediction in advance to plan the observation of an astronomical object. This is where numerical simulation and ML come into play, as they are two widely used approaches for predicting atmospheric variables. 
	- The European Southern Observatory (ESO) is the most important intergovernmental science and technology organization in astronomy, operating in three different sites: La Silla, Paranal, and Chajnantor. Among these, Paranal is the focus of this thesis, where the Very Large Telescope (VLT) is located. Thus, the objective of this work is to predict the average value of seeing one hour into the future, using data from the ESO and an entirely ML-focused framework. 
	- In this thesis, an extensive modification of the database from the ESO has been developed, along with an analysis of it, to generate the best possible results. Additionally, different models have been exhaustively tested to evaluate and select the optimal one in terms of prediction. 
	- The result obtained was 0.151 RMSE and 0.753 Success Rate, reflecting improvements of 14.2% and 4.15%, respectively, compared to the current prediction model. This translates to more accurate predictions along with a higher percentage of observations meeting the requested seeing requirements. Compared to the model currently used at the ESO, this new implementation reflects an average gain of 28 hours per semester for executions that meet the seeing requirements, with 18 hours from the most important observations and 10 hours from those of lower rank. Additionally, the time spent using the telescope for failed executions was reduced by an average of 40 hours per semester. This means more high-quality executions are observed, the telescope is used less for executions that do not meet the required standards, and there is increased capacity to fill the queue with new OBs (Observation Blocks), that is, to conduct new astronomical observations.

---

- <mark>Arevalo-2023</mark>: Sensitivity of Simulated Conditions to Different Parameterization Choices Over Complex Terrain in Central Chile

	- This study evaluates the performance of 14 high-resolution WRF runs with different combinations of parameterizations in simulating the atmospheric conditions over the complex terrain of central Chile during austral winter and spring. 
	- We focus on the **validation of results for coastal, interior valleys, and mountainous areas independently**, and also present an in-depth analysis of two synoptic-scale events that occurred during the study period: a frontal system and a cut-off low. 
	- The performance of the simulations decreases from the coast to higher altitudes, even though the differences are not very clear between the coast and interior valleys for ws10 and precipitation. The simulated vertical profiles show a warmer and drier boundary layer and a cooler and moister free atmosphere than observed. The choice of the land-surface model has the largest positive impact on near-surface variables with the five-layer thermal diffusion scheme showing the smallest errors. 
	- Precipitation is more sensitive to the choice of cumulus parameterizations, with the simplified **Arakawa–Schubert scheme** generally providing the best performance for absolute errors. When examining the performance of the model simulating rain/no-rain events for different thresholds, also the cumulus parameterizations better represented the false alarm ratio (FAR) and the bias score (BS). However, the **Morrison microphysics scheme** resulted in the best critical success index (CSI), while the probability of detection (POD) was better in the simulation without analysis nudging. 
	- **Overall**, these results provide guidance to other researchers and help to identify the best WRF configuration for a specific research or operational goal.

---

- <mark>Kratzert 2019</mark>: Toward Improved Predictions in Ungauged Basins: Exploiting the Power of Machine Learning

	- LSTM networks offer unprecedented accuracy for prediction in ungauged basins. 
	- We trained and tested several LSTMs on 531 basins from the **CAMELS** data set using **k-fold validation**, so that predictions were made in basins that supplied no training data. The training and test data set included  $\sim30$ years of daily rainfall-runoff data from catchments in the US ranging in size from 4 to 2,000 km 2 with aridity index from 0.22 to 5.20, and including 12 of the 13 IGPB vegetated land cover classifications. 
	- This effectively “ungauged” model was benchmarked over a **15-year validation period** against the Sacramento Soil Moisture Accounting (SAC-SMA) model and also against the NOAA National Water Model reanalysis. SAC-SMA was calibrated separately for each basin using 15 years of daily data. The out-of-sample LSTM had higher median Nash-Sutcliffe Efficiencies across the 531 basins (0.69) than either the calibrated SAC-SMA (0.64) or the National Water Model (0.58). 
	- This indicates that there is (typically) sufficient information in available catchment attributes data about similarities and differences between catchment-level rainfall-runoff behaviors to provide out-of-sample simulations that are generally more accurate than current models under ideal (i.e., calibrated) conditions. 
	- We found evidence that adding **physical constraints to the LSTM models might improve simulations**, which we suggest motivates future research related to physics-informed ML.

---

- <mark>Kratzert-2019</mark>: Towards Learning Universal, Regional, and Local Hydrological Behaviors via Machine-Learning Applied to Large-Sample Datasets

	- Regional rainfall-runoff modeling is an old but still mostly out-standing problem in Hydrological Sciences. The problem currently is that traditional hydrological models **degrade significantly in performance when calibrated for multiple basins together instead of for a single basin alone**. 
	- In this paper, we propose a novel, data-driven approach using LSTMs, and demonstrate that under a ’big data’ paradigm, this is not necessarily the case. By training a single LSTM model on 531 basins from the CAMELS data set using **meteorological time series data and static catchment attributes**, we were able to significantly improve performance compared to a set of several different hydrological benchmark models. 
	- Our proposed approach not only significantly outperforms hydrological models that were calibrated regionally but also achieves **better performance than hydrological models that were calibrated for each basin individually**. Furthermore, we propose an adaption to the standard LSTM architecture, which we call an **Entity-Aware-LSTM** (EA-LSTM), that allows for learning, and embedding as a feature layer in a deep learning model, **catchment similarities**. We show that this learned catchment similarity corresponds well with what we would expect from prior hydrological understanding.

---

- <mark>Fernandez 2017</mark>: Sensitivity Analysis of the WRF Model: Wind-Resource Assessment for Complex Terrain

	- Wind energy requires accurate forecasts for adequate integration into the electric grid system. In addition, global atmospheric models are not able to simulate local winds in complex terrain, where wind farms are sometimes placed. For this reason, the use of mesoscale models is vital for estimating wind speed at wind turbine hub height. In this regard, the Weather Research and Forecasting (WRF) Model allows a user to apply different initial and boundary conditions as well as physical parameterizations. In this research, a sensitivity analysis of several physical schemes and initial and boundary conditions was performed for the Alaiz mountain range in the northern Iberian Peninsula, where several wind farms are located. Model performance was evaluated under various atmospheric stabilities and wind speeds. For validation purposes, a mast with anemometers installed at 40, 78, 90, and 118 m above ground level was used. 
	- The results indicate that performance of the Global Forecast System analysis and European Centre for Medium-Range Weather Forecasts interim reanalysis (ERA-Interim) as initial and boundary conditions was similar, although each performed better under certain meteorological conditions. 
	- With regard to physical schemes, there is no single combination of parameterizations that performs best during all weather conditions. Nevertheless, some combinations have been identified as inefficient, and therefore their use is discouraged. As a result, the validation of an ensemble prediction system composed of the best 12 deterministic simulations shows the most accurate results, obtaining relative errors in wind speed forecasts that are ,15%.

---

- <mark>Addor 2018</mark>: Selection of hydrological signatures for large-sample hydrology 

	- Hydrological signatures are now used for a wide range of purposes, including **catchment classification**, process exploration and **hydrological model calibration**. The recent boost in the popularity and number of signatures has however not been accompanied by the development of clear guidance on signature selection, meaning that signature selection is often arbitrary. Here we use three complementary approaches to compare and rank 15 commonly-used signatures, which we evaluate in 671 US catchments from the CAMELS data set (Catchment Attributes and MEteorology for Large-sample Studies). 
	- Firstly, we employ random forests to explore how attributes characterizing the climatic conditions, topography, land cover, soil and geology influence (or not) the signatures. 
	- Secondly, we use a conceptual hydrological model (Sacramento) to critically assess which signatures are well captured by the simulations. 
	- Thirdly, we take advantage of the large sample of CAMELS catchments to characterize the spatial smoothness (using Moran’s I) of the signature field. 
	- These three approaches lead to **remarkably similar rankings of the signatures**. We show that **signatures with the noisiest spatial pattern tend to be poorly captured by hydrological simulations**, that their relationship to catchments attributes are elusive (in particular they are not correlated to climatic indices like aridity) and that they are particularly sensitive to discharge uncertainties. We question the utility and reliability of those signatures in experimental and modeling hydrological studies, and we underscore the general importance of accounting for uncertainties in hydrological signatures.

---

- <mark>Poblete 2020</mark>: Optimization of Hydrologic Response Units (HRUs) Using Gridded Meteorological Data and Spatially Varying Parameters

	- Although complex hydrological models with detailed physics are becoming more common, **lumped and semi-distributed models** are still used for many applications and offer some advantages, such as reduced computational cost. Most of these semi-distributed models use the concept of the hydrological response unit or HRU. In the original conception, HRUs are defined as homogeneous structured elements with similar climate, land use, soil and/or pedotransfer properties, and hence a homogeneous hydrological response under equivalent meteorological forcing. 
	- This work presents a **quantitative methodology**, called hereafter the principal component analysis and hierarchical cluster analysis or **PCA/HCPC** method, to construct HRUs using gridded meteorological data and hydrological parameters. 
	- The PCA/HCPC method is tested using the water evaluation and planning system (WEAP) model for the **Alicahue River Basin**, a small and semi-arid catchment of the **Andes, in Central Chile**. 
	- The results show that **with four HRUs, it is possible to reduce the relative within variance of the catchment up to about 10%, an indicator of the homogeneity of the HRUs**. The evaluation of the simulations shows a good agreement with streamflow observations in the outlet of the catchment with an Nash–Sutcliffe efficiency (NSE) value of **0.79** and also shows the presence of small hydrological extreme areas that generally are neglected due to their relative size.

---

- <mark>Di Giuseppe 2016</mark>: The Potential Predictability of Fire Danger Provided by Numerical Weather Prediction

	- A **global fire danger rating system driven by atmospheric model forcing** has been developed with the aim of providing early warning information to civil protection authorities. The daily predictions of fire danger conditions are based on the U.S. Forest Service National Fire-Danger Rating System (NFDRS), the Canadian Forest Service Fire Weather Index Rating System (FWI), and the Australian McArthur (Mark 5) rating systems. 
	- Weather forcings are provided in real time by the European Centre for Medium-Range Weather Forecasts forecasting system at **25-km resolution**. The global system’s potential predictability is assessed using reanalysis fields as weather forcings. 
	- The Global Fire Emissions Database (**GFED4**) provides **11 yr of observed** **burned areas** from satellite measurements and is used as a validation dataset. 
	- The **fire indices** implemented are **good predictors to highlight dangerous conditions**. High values are correlated with observed fire, and low values correspond to nonobserved events. A more quantitative skill evaluation was performed using the **extremal dependency index**, which is a skill score specifically designed for rare events. It revealed that the three indices were more skillful than the random forecast to detect large fires on a global scale. The performance peaks in the boreal forests, the Mediterranean region, the Amazon rain forests, and Southeast Asia. 
	- The skill scores were then aggregated at the country level to reveal which nations could potentially benefit from the system information to aid decision-making and fire control support. Overall it was found that fire danger modeling based on weather forecasts can provide **reasonable predictability over large parts of the global landmass**

---

- <mark>Carrasco-Escaff 2024</mark>:The key role of extreme weather and climate change in the occurrence of exceptional fire seasons in south-central Chile

	- Unprecedentedly large areas were burned during the 2016/17 and 2022/23 fire seasons in south-central Chile (34-39 ◦ S). These seasonal-aggregated values were **mostly accounted for human-caused wildfires** within a limited period in late January 2017 and early February 2023. 
	- We provide a comprehensive analysis of the meteorological conditions during these events, from **local to hemispheric scales**, and formally **assess the contribution of climate change** to their occurrence. To achieve this, we gathered monthly fire data from the Chilean Forestry Corporation and daily burned area estimates from satellite sources. In-situ and gridded data provided near-surface atmospheric insights, ERA5 reanalysis helped analyze broader wildfire features, high-resolution simulations were used to obtain details of the wind field, and large-ensemble simulations allowed the assessment of climate change’s impact on extreme temperatures during the fires. 
	- This study found extraordinary daily burned area values (>65,000 ha) occurring under extreme surface weather conditions (temperature, humidity, and winds), fostered by strong mid-level subsidence ahead of a ridge and downslope winds converging towards a coastal low. Daytime temperatures and the **water vapor deficit** reached the maximum values observed across the region, well above the previous historical records. We hypothesize that these conditions were crucial in **exacerbating the spread of fire**, along with longer-term atmospheric processes and other non-climatic factors such as fuel availability and increasing human-driven ignitions. 
	- Our findings further reveal that climate change has increased the probability and intensity of extremely warm temperatures in south-central Chile, underscoring anthropogenic forcing as a significant driver of the extreme fire activity in the region.

---

- <mark>Alet 2025</mark>: Skillful joint probabilistic weather forecasting from marginals

	- Machine learning (ML)-based weather models have rapidly risen to prominence due to their greater accuracy and speed than traditional forecasts based on numerical weather prediction (NWP), recently outperforming traditional ensembles in global probabilistic weather forecasting. This paper presents FGN, a simple, scalable and flexible modeling approach which significantly outperforms the current state-of-the-art models. FGN generates ensembles via learned model-perturbations with an ensemble of appropriately constrained models. It is trained directly to minimize the continuous rank probability score (CRPS) of per-location forecasts. 
	- It produces **state-of-the-art ensemble forecasts** as measured by a range of deterministic and probabilistic metrics, makes skillful ensemble tropical cyclone trackpredictions, and captures joint spatial structure despite being trained only on marginals.
---





[1]: Not a Paper 
[2]: Prepared for Submission to JCAP (_Journal of Cosmology and Astroparticle Physics_)