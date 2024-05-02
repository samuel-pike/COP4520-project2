# COP4520-project2

Second course project for COP4520 - Computing in Massively Parallel Systems

This project is about computing spatial distance histogram (SDH) of a collection of 3D points. The SDH problem can be formally described as follows: given the coordinates of N particles (e.g., atoms, stars, moving objects in different applications) and a user-defined distance w, we need to compute the number of particle-to-particle distances falling into a series of ranges (named buckets) of width w: [0, w), [w, 2w), . . . , [(l − 1)w, lw]. Essentially, the SDH provides an ordered list of non-negative integers H = (h0, h1, . . . , hl−1), where each hi(0 ≤ i < l) is the number of distances falling into the bucket [iw, (i + 1)w). Given a bucket width w, this project aims to compute and output the quantity of particles in each bucket.

I have written a CUDA kernel that:
* Transfers the input data array onto GPU device using CUDA.
* Computes the distance between one point to all other points and updates the histogram accordingly.
* Copies the final histogram from the GPU back to the host side.
* Compares this histogram with the computed CPU histogram bucket by bucket.

When ran on the USF GAIVI cluster, for 10000 particles and bucket width of 500,
* **1.3527 s** runtime on CPU
* **0.0734 s** runtime on GPU 

After creating this CUDA kernel, I applied the following optimizations:
* Utilized shared memory to optimize communication between blocks.
* Coalesced memory accesses such that each warp accesses consecutive memory addresses when possible.
* Applied compiler-level optimizations to improve loop performance per thread.

When ran on the USF GAIVI cluster, for 10000 particles and bucket width of 500,
* **0.0743 s** runtime on unoptimized kernel
* **0.0167 s** runtime on optimized kernel

Overall, the program is able to achieve an **81x** improvement to runtime over the CPU algorithm.
