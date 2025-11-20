# Simulations: Ising/liquid-vapor model, GPU

- <mark>Lipowski 2023</mark>: Heat-Bath and Metropolis Dynamics in Ising-like Models on Directed Regular Random Graphs

	- Using a **single-site** **mean-field approximation** (MFA) and Monte Carlo simulations, we examine **Ising-like** models on **directed regular random graphs**. The models are directed-network implementations of the Ising model, Ising model with absorbing states, and majority voter models. 
	- When these nonequilibrium models are driven by the heat-bath dynamics, their stationary characteristics, such as magnetization, are correctly reproduced by MFA as confirmed by Monte Carlo simulations. It turns out that **MFA reproduces the same result as the generating functional analysis** that is expected to provide the exact description of such models. 
	- We argue that on directed regular random graphs, the neighbors of a given vertex are typically uncorrelated, and that is why MFA for models with heat-bath dynamics provides their exact description. 
	- For models with **Metropolis dynamics**, certain additional correlations become relevant, and MFA, which neglects these correlations, is **less accurate**. 
	- Models with heat-bath dynamics undergo **continuous phase transition**, and at the critical point, the power-law time decay of the order parameter exhibits the behavior of the Ising mean-field universality class. 
	- Analogous phase transitions for models with Metropolis dynamics are **discontinuous**

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

- <mark>Bauerschmidt 2024</mark>: Kawasaki dynamics beyond the uniqueness threshold

	- Glauber dynamics of the Ising model on a random regular graph is known to mix fast below the tree uniqueness threshold and exponentially slowly above it. We show that **Kawasaki dynamics of the canonical ferromagnetic Ising model** on a random d-regular graph mixes fast beyond the tree uniqueness threshold when $d$ is large enough (and conjecture that it mixes fast up to the tree reconstruction threshold for all $d \geq 3$). This result follows from a more general spectral condition for (modified) **log-Sobolev inequalities for conservative dynamics of Ising models**. The proof of this condition in fact extends to perturbations of distributions with **log-concave generating polynomial**.

---

- <mark>Lee 2024</mark>: Parallelising Glauber dynamics

	- For distributions over discrete product spaces $\Pi_{i=1}^n\Omega_{i}'$, Glauber dynamics is a Markov chain that at each step, resamples a random coordinate conditioned on the other coordinates. 
	- We show that k-Glauber dynamics, which resamples a random subset of k coordinates, mixes k times faster in $\chi^2$-divergence, and assuming approximate tensorization of entropy, mixes k times faster in KL-divergence. We apply this to obtain parallel algorithms in two settings: 
		- (1) For the Ising model $µ_{J,h} (x) \propto exp(\frac{1}{2} \langle x, Jx\rangle + \langle h, x \rangle)$ with $||J|| < 1 − c$ (the regime where fast mixing is known), we show that we can implement each step of $\tilde{\Theta}(n/|J|_F)$-Glauber dynamics efficiently with a parallel algorithm, resulting in a parallel algorithm with running time $\tilde{O}(||J||_F) = \tilde{O}(\sqrt{n})$. 
		- (2) For the mixed p-spin model at high enough temperature, we show that with high probability we can implement each step of $\tilde{\Theta}(\sqrt{n})$-Glauber dynamics efficiently and obtain running time $\tilde{O}(\sqrt{n})$

---

- <mark>Romero 2019</mark>: A Performance study of the 2D Ising Model on GPU 

	- The simulation of the 2D Ising model is used as a **benchmark to show the computational capabilities of GPUs**. The rich programming environment now available on GPUs and flexible hardware capabilities allowed us to quickly experiment with several implementation ideas: a simple stencil-based algorithm, recasting the stencil operations into matrix multiplies to take advantage of Tensor Cores available on NVIDIA GPUs, and a highly optimized multi-spin coding approach. Using the managed memory API available in CUDA allows for simple and efficient distribution of these implementations across a multi-GPU NVIDIA DGX-2 server. 
	- We show that even a **basic GPU implementation can outperform current results published on TPUs** and that the optimized multi-GPU implementation can simulate very large lattices **faster than custom FPGA solutions**

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

	- This work is devoted to the consistent modeling of a three-phase mixture of a gas, a liquid and its vapor. Since the gas and the vapor are miscible, the mixture is subjected to a non-symmetric constraint on the volume. Adopting the **Gibbs formalism**, the study of the extensive equilibrium entropy of the system allows to recover the Dalton’s law between the two gaseous phases. In addition, we distinguish whether phase transition occurs or not between the liquid and its vapor. The thermodynamical equilibria are described both in extensive and intensive variables. In the latter case, we focus on the geometrical properties of equilibrium entropy. The consistent characterization of the thermodynamics of the three-phase mixture is used to introduce two Homogeneous Equilibrium Models (HEM) depending on mass transfer is taking into account or not. Hyperbolicity is investigated while analyzing the entropy structure of the systems. 
	- Finally we propose two Homogeneous Relaxation Models (HRM) for the three-phase mixtures with and without phase transition. Supplementary equations on mass, volume and energy fractions are considered with appropriate source terms which model the relaxation towards the thermodynamical equilibrium, in agreement with entropy growth criterion.

---

- <mark>Burgos 2023</mark>: A Deep Potential model for liquid-vapor equilibrium and cavitation rates of water

	- Computational studies of liquid water and its phase transition into vapor have traditionally been performed using classical water models. Here we utilize the Deep Potential methodology —a ML approach— to study this ubiquitous phase transition, starting from the phase diagram in the liquid-vapor coexistence regime. The machine learning model is trained on ab initio energies and forces based on the SCAN density functional which has been previously shown to reproduce solid phases and other properties of water. Here, we compute the surface tension, saturation pressure and enthalpy of vaporization for a range of temperatures spanning from 300 to 600 K, and evaluate the Deep Potential model performance against experimental results and the semi-empirical TIP4P/2005 classical model. Moreover, by employing the seeding technique, we evaluate the free energy barrier and nucleation rate at negative pressures for the isotherm of 296.4 K. We find that the nucleation rates obtained from the Deep Potential model deviate from those computed for the TIP4P/2005 water model, due to an underestimation in the surface tension from the Deep Potential model. 
	- From analysis of the seeding simulations, we also evaluate the **Tolman length** for the Deep Potential water model, which is ($0.091 \pm 0.008$) nm at 296.4 K. Lastly, we identify that water molecules display a preferential orientation in the liquid-vapor interface, in which H atoms tend to point towards the vapor phase to maximize the enthalpic gain of interfacial molecules. We find that this behaviour is more pronounced for planar interfaces than for the curved interfaces in bubbles. 
	- This work represents the first application of Deep Potential models to the study of liquid-vapor coexistence and water cavitation.

---

- <mark>Jha 2023</mark>: GPU-acceleration of tensor renormalization with PyTorch using CUDA

	- We show that numerical computations based on tensor renormalization group (TRG) methods can be signiﬁcantly accelerated with PyTorch on graphics processing units (GPUs) by leveraging NVIDIA’s Compute Uniﬁed Device Architecture (CUDA). We ﬁnd improvement in the runtime (for a given accuracy) and its scaling with bond dimension for two-dimensional systems. Our results establish that utilization of GPU resources is essential for future precision computations with TRG.

---

- <mark>Jin 2025</mark>: Supercritical fluids as a distinct state of matter characterized by sub-short-range structural order

	- A supercritical fluid (SCF) – the state of matter at temperatures and pressures above the critical point – exhibits properties intermediate between those of a liquid and a gas. However, whether it constitutes a fundamentally distinct phase or merely a continuous extension of the liquid and gas states remains an open question. Here we show that a **SCF is defined by sub-short-range** (SSR) **structural order** in the spatial arrangement of particles, setting it apart from the gas (disordered), liquid (short-range ordered), and solid (long-range ordered) states. 
	- The SSR structural order can be characterized by a length scale effectively quantified by the number of observable peaks in the radial distribution function. This length grows from a minimum microscopic value, on the order of the inter-particle distance at the gas–SCF boundary, to a diverging value at the SCF–liquid boundary. Based on the emergence of SSR order, we demonstrate that the transport and dynamical properties of the SCF state, including the diffusion coefficient, shear viscosity, and velocity autocorrelation function, also clearly distinguish it from both the liquid and gas states. 
	- **Theoretical predictions are validated by molecular dynamics simulations** of argon and further supported by existing experimental evidence. Our study confirms and assigns physical significance to the refined phase diagram of matter in the supercritical region, consisting of **three distinct states** (gas, supercritical fluid, and liquid) separated by two crossover boundaries that follow universal scaling laws.

---

- <mark>CHERRINGTON 2008</mark>: A DUAL ALGORITHM FOR NON-ABELIAN YANG-MILLS COUPLED TO DYNAMICAL FERMIONS

	- We extend the dual algorithm recently described for pure, non-abelian Yang-Mills on the lattice to the case of lattice fermions coupled to Yang-Mills, by constructing an ergodic Metropolis algorithm for dynamic fermions that is local, exact, and built from gauge-invariant boson-fermion coupled configurations. Fon concreteness, we present in detail the case of 3D, for the group SU(2) and staggered fermions, however the algorithm readily generalizes with regard to group and dimension. 
	- The treatment of the fermion determinant makes use of a polymer expansion; as with previous proposals making use of the polymer expansion in higher than 2D, the critical question for practical applications is whether the presence of negative amplitudes can be managed in the continuum limit.

---






