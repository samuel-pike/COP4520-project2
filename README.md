# COP4520-project2

Second course project for COP4520 - Computing in Massively Parallel Systems

This project is about computing spatial distance histogram (SDH) of a collection of 3D points. The SDH problem can be formally described as follows: given the coordinates of N particles (e.g., atoms, stars, moving objects in different applications) and a user-defined distance w, we need to compute the number of particle-to-particle distances falling into a series of ranges (named buckets) of width w: [0, w), [w, 2w), . . . , [(l − 1)w, lw]. Essentially, the SDH provides an ordered list of non-negative integers H = (h0, h1, . . . , hl−1), where each hi(0 ≤ i < l) is the number of distances falling into the bucket [iw, (i + 1)w). Clearly, the bucket width w is a key parameter of an SDH to be computed.

I have written a CUDA kernel that:
* Transfers the input data array onto GPU device using CUDA
* Computes the distance between one point to all other points and updates the histogram accordingly.
* Copies the final histogram from the GPU back to the host side
* Compares this histogram with the computed CPU histogram bucket by bucket.

