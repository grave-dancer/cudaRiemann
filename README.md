# cudaRiemann
Simple Riemann sums for CUDA.

The number of threads is the accuracy. 
You can use more threads with more blocks to get greater integration accuracy and faster speed over larger intervals.
By default only one block with 255 threads is used, to increase parallelism.
