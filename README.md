# AB_Tiling

This repository contains code related to two main aspects of my PhD research on the Ammann–Beenker (AB) tiling:

- Construction of **Hamiltonian cycles** using our algorithm described in https://journals.aps.org/prx/abstract/10.1103/PhysRevX.14.031005.
- Simulation and analysis of the **Fully Packed Loop (FPL)** model on the AB tiling.

These tools were developed for use in my thesis and related publications, including our PRX paper.

---

## AB Tiling Construction

1. **Geometry_sp.py**  
   Generates the adjacency matrix for the AB tiling. Originally written by our collaborator Jérôme Lloyd for his PRB paper with Felix, I later modified this code for our PRX paper to include the `decorate()` function. This function decorates inflated AB prototiles with dimers, which eventually join to form full Hamiltonian cycles.

2. **ContractingNodes.py**  
   Contracts nearby nodes to create a connected AB tiling graph.

3. **plaquette.py**  
   Identifies and classifies plaquettes (elementary loops) in any input lattice.

4. **Hcycle_d0+d1+d2copy.py**  
   Constructs Hamiltonian cycles within the \( U_2 \) region of the AB tiling.

5. **Figures.py**  
   Generates publication-quality figures used in our PRX paper.

---

## Fully Packed Loop (FPL) Model

1. **Final_FPLsNew.py**  
   Generates FPL configurations using local flip dynamics.

2. **FPLs_C.py**  
   Lists and analyzes loops in a given FPL configuration.

3. **Heights.py**  
   Computes height mappings for the FPL model on the AB tiling.

4. **RoG.py**  
   Calculates the Radius of Gyration \( R \) vs loop length, and estimates loop area distributions \( P(s) \) vs loop length \( s \).

5. **FPLs_Cr.py**  
   Computes the two-point correlation function \( C(r) \) for the FPL model on AB tiling.
