# MSM Algorithm Optimization: the Pippenger algorithm Based on Preprocessing Technique(elastic MSM)
## 1. Introduction
Currently, most of the computational overhead in ZKP systems comes from local computation. Furthermore, the succinctness of proof systems typically requires verification time to be sublinear (or even logarithmic), which means the Prover incurs a significant time cost in generating proofs. As a result, modern zk-SNARK protocols often see a scenario where the Verifier can verify a proof in just one second, but the Prover may need a much longer time to generate the proof.

Looking at the construction of ZKP systems, we can see that hardware costs primarily concentrate on MSM, NTT, and arithmetic hashing.

**MSM:** There are many ways to optimize MSM. For large-scale MSM, algorithms like Pippenger can be used to reduce computational complexity (originally linear, it can be reduced to $O(n/log_n))$. Alternatively, using special points or curves can reduce the computational cost of each field operation.

**NTT:** The typical algorithm is Cooley-Tukey, with a complexity of $O(n\log_n)$. However, certain characteristics of NTT prevent it from being easily parallelized. (NTT computations have inherent dependencies, making it difficult to split into subproblems and merge them later.) Furthermore, NTT calculations require all elements involved to be stored in memory, which means larger-scale NTT computations require larger memory allocations.

**Arithmetic Hashing:** Arithmetic hashing is a widely used cryptographic primitive in ZKP, particularly in constructions based on Merkle Trees. Since hashes are generally very short, they are well-suited for representing large datasets. Common arithmetic hashes used in ZKP include Poseidon, Blake3, and Poseidon2. These hashes incur significant costs during local computation, and their computational efficiency is influenced by factors such as field size, prime numbers used, protocol rounds, and MDS matrix structure. 

This paper focuses on a new MSM acceleration algorithm, called elastic MSM, inspired by the work presented in reference <a href="#ref1">[1]</a>.


## 2. MSM

The MSM calculation can be represented by the following equation:

$$Q=\sum_{i=1}^n{k_i}\cdot P_i $$

where $(P_1, ..., P_n)$ are $n$ group elements,  $\lbrace k_i\rbrace _{i∈[n]}$ are $\lambda$-bit scalars. The range of $\lambda$ can be between 254 and 768, while $n$ is on the order of millions. ${P_i}$ are points on an elliptic curve.
Therefore, MSM primarily involves the inner product operation of elliptic curve vectors: point addition (PADD) and Therefore, MSM primarily involves the inner product operation of elliptic curve vectors: point addition (PADD) and point multiplication (PMULT).
 
## 2. Pip­penger 
Pippenger's algorithm, proposed in 1976，breaks down the MSM calculation into three main steps:

**(1) Splitting**

* Subtask Division: The original task is divided into smaller subtasks. A window of size $s$ is defined. Each $λ$-bit scalar $k_i$ is split into $s/λ$ parts, each consisting of an $s$-bit scalar denoted as $m_{ij}​$, representing the $j$-th sub-scalar of $k_i$. For the divided sub-scalar $m_{ij}$, the following equation holds for any scalar $k_i$:

$$k_i=\sum_{j=1}^{\lambda/s}{2^{(j-1)s}}\cdot m_{ij} $$

* Transformation: Based on the sub-scalars $m _{ij}​$, the calculation for the $i$-th group element $P_i$can be transformed into:

$$G_j=\sum_{i=1}^{n} m_{ij}\cdot P_i $$

* MSM Transformation: Therefore, the MSM equation $Q=\sum_{i=1}^n{k_i}\cdot P_i$ can be rewritten as:

$$Q=\sum_{i=1}^{n}Q_i=\sum_{i=1}^{n}k_i\cdot P_i\\ =\sum_{i=1}^{n}\sum_{j=1}^{\lambda/s} (2^{(j-1)s}\cdot m_{ij})\cdot P_i \\=\sum_{j=1}^{\lambda/s} 2^{(j-1)s}\cdot \sum_{i=1}^{n}m_{ij}\cdot P_i\\=\sum_{j=1}^{\lambda/s} 2^{(j-1)s}\cdot G_j$$

**(2) Block Computation**

* Scalar Multiplication: Calculate the scalar multiplication for each subtask, i.e., compute each $G_j(j\in [1,\lambda/s])$.

**(3) Summation**

* Weighted Sum: Multiply the summation results of each subtask by a power coefficient, dependent on the window size $s$, and sum them together. The summation process is as follows:

$$Q=\sum_{j=1}^{\lambda/s} 2^{(j-1)s}\cdot G_j\\=2^s(\cdots (2^s(s^s\cdot G_{\lambda /s}+G_{\lambda /s -1})+G_{\lambda /s -2})\cdots)+G_1$$

Figure 1 illustrates the implementation of the Pip­penger algorithm.

![alt text](<fig_MSM\Pippenge.png>)

<p align="center"><font face="黑体" size=3.>Figure 1 Pippenger</font></p>

In the figure, $\lambda =12$,$s=4$, $j=3$. $2^s-1=15$ chunks are created, and related $P_i$ are placed in corresponding chunks. The chunks are labeled as $B_1,\cdots,B_{15}$.

The sub-scalars for $P_1​,…,P_5$ (the three boxes in the middle left of the figure) are initially represented as binary numbers. For example, the red box (1101) corresponds to group elements $P_1 and $P_5​$, so $P_1​$ and $P_5​$ are placed in the bucket $B_{13}$ (the binary representation of 13 is 1101). The remaining sub-scalars are processed similarly. $P_2$ is placed in $B_{14}$, and $P_3$, $P_4$ are placed in $B_{15}$. This process is represented by the red part in the upper right corner of the figure. The green and blue parts represent the processing of the remaining sub-scalars.

The group elements in the same bucket are then added together (PADD). For example, the red part $B _{13}$ actually contains the result of $P_1+P_5$.

The sum of the chunks for the $j$-th part (using the red part as an example): $G_3=13⋅B _{13}+14⋅B_{14}+15⋅B_{15}=13⋅(P_1+P_5)+14⋅P_2+15⋅(P_3+P_4)$

Finally, $Q=2^8⋅G_3​+2^4⋅G_2+G_1$.

## 3. elastic MSM

The core of the elastic MSM algorithm is to further split the computation process based on Pippenger's algorithm. The number $\lambda/s$ in

$$Q=\sum_{i=1}^{n}Q_i=\sum_{i=1}^{n}k_i\cdot P_i\\ =\sum_{i=1}^{n}\sum_{j=1}^{\lambda/s} (2^{(j-1)s}\cdot m_{ij})\cdot P_i$$

is split into two integers, so that $\lambda/s=\omega\cdot k$. Therefore, we have:

$$Q_i=\left[\begin{matrix}1&2^s&2^{2s}&\cdots&2^{(\omega k-1)s}\end{matrix}\right]\cdot \left[\begin{matrix}m_{i1}P_i\\m_{i2}P_i\\m_{i3}P_i\\\vdots\\m_{i(\omega k)}P_i\end{matrix}\right]$$
​
We define a new notation $M_{i(l,t)}=m_{i((l-1)k+t)}$, where $l\leq \omega$, $t\leq k$. This leads to:

$$ Q_i= \sum _{j=1}^{\omega k} (2^{(j-1)s}\cdot m_{ij})\cdot P_i\\=\sum_{l=1}^{\omega}\sum_{t=1}^k{2^{((l-1)k+(t-1))s}\cdot M_{i(l,t)}\cdot P_i}$$

Based on the above equation, the calculation of $Q_i$ is transformed into the following matrix multiplication:

$$Q_i=\left[\begin{matrix}1&2^{ks}&\cdots&2^{(\omega -1)ks}\end{matrix}\right]\cdot P_i \cdot \left[\begin{matrix}m_{i(1,1)}&m_{i(1,2)}&\cdots&m_{i(1,k)}\\m_{i(2,1)}&m_{i(2,2)}&\cdots&m_{i(2,k)}\\\vdots&\vdots&\vdots&\vdots\\m_{i(\omega, 1)}&m_{i(\omega, 2)}&\cdots&m_{i(\omega, k)}\end{matrix}\right]\left[\begin{matrix}1\\2^s\\\vdots\\2^{(k-1)s}\end{matrix}\right]$$

Next, we define three auxiliary variables:

$$P_{ij}=2^{(j-1)ks}\cdot P_i $$
$$G_{il}=\sum_{j=1}^\omega M_{i(j,l)}\cdot P_{ij}$$
$$N_{ij}=\sum_{l=1}^k 2^{(l-1)s}\cdot M_{i(j,l)}$$

The three auxiliary variables have the following meanings:

* $P_{ij}$：This represents the result of multiplying $P_i$ by a power of 2. Specifically, it is the product of $P_i$ and the $j$-th element of the row vector $\left[\begin{matrix}1&2^{ks}&\cdots&2^{(\omega -1)ks}\end{matrix}\right].

* $G_{il}$：This is the sum of the products of the elements in the $l$-th column of the $M$ matrix with$P_{ij}$.

* $N_{ij}$：This is the product of the $j$-th row of the $M$ matrix with the column vector $\left[\begin{matrix}1&2^{s}&\cdots&2^{(k -1)s}\end{matrix}\right]^\mathbf{T}$.

Based on these three auxiliary variables, $Q_i$ can be rewritten as:

$$Q_i = \sum_{j=1}^{\omega} N_{ij} \cdot P_{ij}
= \sum_{j=1}^{\omega} \sum_{l=1}^{k} {2^{(l-1)s}}\cdot M_{i(j,l)} \cdot P_{ij} \\
= \sum_{l=1}^{k}2^{(l-1)s }\cdot \sum_{j=1}^{\omega} \cdot M_{i(j,l)} \cdot P_{ij}
= \sum_{l=1}^{k}2^{(l-1)s } G_{il}$$

Finally, summing up all $Q_i$ yields the original result:

$$Q=\sum_{i=1}^n Q_i$$
​
The following diagram illustrates the specific implementation of the Elastic Pippenger algorithm.

![alt text](<fig_MSM\Elastic Pippenger.png>)

<p align="center"><font face="黑体" size=3.>Figure 1 Elastic Pippenger</font></p>

As the diagram shows, the core of Elastic Pippenger is to add a pre-processing step based on the original Pippenger algorithm. This step decomposes $P_i$ into $P_{ij}$ and performs pre-computation:

$$P_{ij}=2^{(j-1)ks}\cdot P_i $$

## 4. Parallel Computing for Pip­penger

Three methods for parallelizing calculations are presented:
* Approach 1: Scalar Coefficient Decomposition

Based on the formula  $Q=\sum_{j=1}^{\lambda/s} 2^{(j-1)s}\cdot G_j$, the simplest approach for parallelizing Pippenger is to compute the corresponding $G_j$ for each of the $s/λ$ tasks in parallel. Then, the results of each task are multiplied by the corresponding coefficient $2^{(j−1)s}$ and summed up.

This method achieves a speedup of at most $s/λ$. In practice, the value of $λ$ is determined by the chosen elliptic curve. Even with a reasonable choice of block size $s$, the acceleration effect is limited and cannot fully utilize the computational power of GPUs.

* Approach 2: Group Element Decomposition

Assuming the number of group elements $n$ is divisible by $T$, the equation$Q=\sum_{i=1}^n{k_i}\cdot P_i$ is split into $T$ subtasks $Q_t​,t \leq T$. The final result is obtained by merging the computed results of each subtask: $Q=\sum _{t=1}^T{Q_t}$

This method allows for parallel processing of a sufficient number of tasks on the GPU, as long as the size of the divided tasks is reasonable, until the GPU's computational power is fully utilized.

* Approach 3: Double Decomposition:

This method combines the above two methods by dividing both into $s/λ$ $G_j$ and $t$ subtasks.

$$Q=\sum_{i=1}^n{k_i}\cdot P_i =\sum_{j=1}^{T/(\lambda/s)}\cdot Q_j\\=\sum_{j=1}^{T/(\lambda/s)}\sum_{l=1}^{\lambda/s} 2^{(l-1)s}\cdot G_{jl}$$

This approach allows for a more flexible and efficient parallelization strategy, leveraging the strengths of both scalar and group element decomposition methods.

## 5.  Evaluation

Performance on two parallel Pippenger algorithm: Approach 2 and Approach 3.

Table 1 and Table 2 provide the evaluation results on the precomputing time and MSM time of *fast parallel Pippenger*, *GZKP Pippenger* and *elastic Pippenger* across various preprocessing storage space limitations. Given a range of practical parameters, it can be concluded from these tables that the *elastic Pippenger* has a slight advantage in precomputing time compared to *GZKP Pippenger* across all preprocessing storage space limitations.

Notably, the precomputing time of *GZKP
Pippenger* (Approach 2) or *elastic Pippenger* (Approach 2) is the same as the precomputing time of *GZKP
Pippenger* (Approach 3) or *elastic Pippenger* (Approach 3). This is because the only difference lies in the use of different MSM algorithms, but the preprocessing methods are consistent.
<p align="center"><font face="黑体" size=3.>Table 1  Performance results about approach 2

</font></p>
![alt text](<fig_MSM\Table 1.png>)

<p align="center"><font face="黑体" size=3.>Table 2  Performance results about approach 3 

</font></p>
![alt text](<fig_MSM\Table 2.png>)

When the storage space is limited to storing $7 · 2^{22} − 2^{22}$ extra points, the elastic Pippenger (Approach 2) achieves about 56% − 90% speedup versus fast parallel Pippenger (Approach 2) and the elastic
Pippenger (Approach 3) achieves about 2% − 8% speedup versus fast parallel Pippenger (Approach 3). Thus, the fast parallel Pippenger (Approach 2) could be optimized more by preprocessing techniques. 

<p name = "ref1">[1]Xudong Z.,Haoqi H.Zhengbang Y.,et al. Elastic MSM: A Fast, Elastic and Modular
Preprocessing Technique for Multi-Scalar
Multiplication Algorithm on GPUs. https://eprint.iacr.org/2024/057.pdf