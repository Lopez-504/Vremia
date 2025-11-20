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






