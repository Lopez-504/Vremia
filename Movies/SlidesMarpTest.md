---
theme: academic
paginate: true
math: latex
header: Structure formations and quasi-sperical collapse
footer: LFIS 421
---

<!-- _header: Title -->
# Formación de estructuras y colapso quasi-esférico en $\Lambda$CDM
![bg right:32%](../uni1.jpg)
Jorge López Retamal   |   2025-11-30

Licenciatura en Física mención Computación Científica, **Universidad de Valparaíso**.

---
<!-- _header: table of contents -->
![bg right:10%](../uni1.jpg)
1. Introducción a Relatividad numérica
2. Condiciones iniciales
3. Distribuciones iniciales
4. Evolución lineal 
5. Introduce the Top-Hat model and Eloisa's work for comparison
6. Non-linear evolution, study and comparison of collapse
7. Conclusions
---
# Introduction to numerical relativity

- N+1 or ADM formalism
	- Lapse, shift, fluid velocity, expansion tensor
	- Synchronous comoving gauge
	- Not strongly hyperbolic system system of equations and the need of a conformal trasnformation
---
<!-- backgroundColor: #FFFCF2 -->
- For the lapse, shift, metric and so on, we go from **ADM** to **BSNN** 
- For the matter variables we go from primitive variables to conserved variables (Valencia formulation)
- These are the ones that **evolve** in the simulation, but then get translated back to ADM and dust perfect fluid primitive variables for our data analysis
- We monitor our simulation with the momentum and Hamiltonian constraint (obtained from Einstein's equations)
---
- If we're using an FLRW metric, our equations are the Friedman equations and the continuity equation, but If we want to consider a perturbed FLRW metric, we should use these more general equations:
		1. Momentum constraint  
		2. Hamiltonian constraint 
		3. Raychaudhuri eq
		4. Conservation eq

---
Initial conditions

- We use the derivation of [[Non-Gaussian Initial Contitions in LambdaCDM]]
- Start with a **flat FLRW metric and fully perturb the metric and the energy density** (with the density contrast $\delta$)
- But due to our gauge choice we cancel out temporal perturbations
- We also cancel out vector and tensor perturbations because from **inflationary models**, they're expected to be subdominant during the matter dominated era
- Then we only consider scalar perturbations
---
- We perturb the metric with $\Psi$ and $\chi$, then we obtain the $^{(3)}R$ which can be written in terms of $R_c$
- This is relevant because $R_c$ and $\zeta^{(1)}$ are used to quantify perturbations created during inflation
- $R_c$ is gauge invariant at first order and time-independent, i.e., a great variable to work with
---
- What they do in this article is express the scalar perturbations $\Psi$, $\chi$ and $\delta$ as functions of the curvature perturbation $R_c$, then all we have to do is define $R_c$ to completely express our system
	- They start from the evolution of the density contrast $\delta^{(1)}$, and for its diff. eq. we have two solutions: 
	- The homogeneous solution corresponds to the **decaying mode**, related to Hubble expansion $\delta_-\propto H$   
	- The particular solution corresponds to the growing mode, related to the curvature perturbation $R_c$
---
- But in the matter dominated era, the decaying mode is expected to have fully decayed away, so we only consider the particular solution (growing mode $\delta_+$) 
- Thus, we express $\delta$ as a function of $R_c$
	- Finally, we use the continuity eq and the Hamiltonian eq to express $\Psi$ and $\chi$ as functions of $R_c$
		- With this, we can express our perturbed metric $\gamma_{ij}$, up to first order, in terms of $R_c$ and from it derive the extrinsic curvature $K_{ij}$
			- The other parameters needed to express $\gamma_{ij}$ and $K_{ij}$ are obtained from the $\Lambda$CDM model and we define those using the *Plank 2018* data ($h$, $\Omega_{m0}$)
---
<!-- _header: > -->
- We can do the same with the matter density, however we use the **Hamiltonian constraint** 
$$\rho=\frac{1}{2\kappa}\left(^{(3)}R+\frac{2}{3}K^2-2A^2-2\Lambda\right)$$
where all these quantities are **computed in full from the 1st order expressions**, giving us the matter density $\rho$ **in full**  
- All that's left to do is to define $R_c$. We choose to defined it as a 3D sinusoidal
$$R_c=A_\text{pert}\sum \sin\left(\frac{2\pi x^i}{\lambda_\text{pert}}\right)$$
where the amplitude $A_\text{pert}$ and the wavelegth $\lambda_\text{pert}$ as free parameters. 
- The initial redshift remains free however to be consistent with our assumptions on the decaying mode, we make sure to reamain in the matter dominated era

---
Initial distribution

- Show isocurves of the density contrast
	- Show spherical symmetry near OD and UD, but filamentary structures as we move away from them, that's why we call this a quasi-spherical distribution
	- Present slice plots for $\delta$, $\delta \gamma$, $\delta K$ and $^{(3)}R$ to show that initially all this quantities reflect the same structre of $R_c$ 
	- Present plots to visualize the impact of the free parameters in the initial distribution

---
Evolution
|![fig5](../fig5.png)|
|:--:| 
| *Fig 5: caption here* |



---
Linear evolution

Solo en caso de obtener resultados en la simulación sim64_linA

---
<!-- _header:  -->
- Introduce the Top-Hat model and Eloisa's work for comparison
	- The Top-Hat model describes a perfectly spherically symmetric OD in a background universe
		- Initially, the radius will expand with the expansion of the universe, but eventually it'll reach a maximum radius (TA point) and after that the radius will decrease and the OD will virialize/collapse
		- This model gives density contrast values for the TA and virialization points (cite Vitorio), so we'll see if we get similar values in our simulations
---
- Eloisa and Marco in (cite) do something similar but instead of defining $R_c$ they define $\delta_\text{IN}$ as a 3D sinusoidal and set the initial extrinsic curvature to a constant equal to the background value 

$$K_{\text{IN }ij}=\bar{K}_{ij}$$

- Then, to get $\gamma_{ij}$ they solve the hamiltonian constraint using and elliptic solver (`CT_MultiLevel` written by Eloisa)
- They introduce both a **growing** and **decaying mode**


---
Non-linear evolution, study and comparison of collapse

- Present perturbation evolution during collapse (Fig 5) for each place: OD, UD and CT
		- The sign change in the $\delta\gamma$ plot happens because we start at the curvature dominated regime (also called **long wavelenght regime**)
		- For the $\delta K$ plot, we see how the OD expands with the universe at the begining but then slows down and reaches TA, then it starts collapsing inwards
			- For the UD we see that the expansion is faster than the background
			- We can't really say if it converges to the Milne model (in that model there's no energy density)
---
- We can identify open and closed universes by looking at the $a^2$$^{(3)}R$ plot
			- In the UD we can see that it starts as an open universe but then it tends towards a flat one
	- Table to compare our simulations with the Top-Hat model and Eloisa's results
		- We use the turn around and virialization points for comparison
		- The variables are the normalized scale factor and the density contrast (both in full and at 1st order)
			- Notice that our simulation cannot portrait virialization since it's a hydrodynamical simulation, instead we just see that our simulation crashes
---
- From this we coclude that 
	1. The decaying mode (present in Eloisa's work) **slows down the virialization** 
	2. The Top-Hat model is a **good approximation at the peak of the OD**
- Evolution of the infalling region
	- Analyze different directions
	- Fig 7: slice of the trace of the extrinsic curvature
		- We observe the TA boundary, dividing the expanding (outside) region from the infalling (inside) region
		- We observe how we go from a rather spherical profile to a distict shape where the direction does matter (the octahedron)
---
- Contributions to the Raychoudhuri equation
	- Fig 6: At the TA and right before the crash
		- If we cancel out the shear in the Hamiltonian constraint and the Raychudhuri eq, we become independent of the surrounding enviroment, and we retrieve the Friedman equations
			- This is why the peak of the OD matches the Top-Hat model

