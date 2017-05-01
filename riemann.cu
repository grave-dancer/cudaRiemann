#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>

//Shared variable
__shared__ float sum;

__device__ float f(float x)
{
    return 2*x; //The function
}

__device__ float trapezoid(float x1, float x2, float y1, float y2)
{
    //Calculate area of trapezoids
    if (y1 > y2) return ((y1*abs(x1-x2))) - (((y1-y2)*abs(x1-x2))/2.0f);
    if (y2 > y1) return ((y2*abs(x1-x2))) - (((y2-y1)*abs(x1-x2))/2.0f);
    return ((y1*abs(x1-x2))); //rectangle case

}

__global__ void riemannSum(
        float* interval1,
        float* interval2,
        float* totalThreads,
        float* output)
{



            float ind;
            
            //Scale the threads to match the interval
            //Every trapezoid has width (interval2 - interval1)/blockDim.x
            ind = ((abs(interval2[0] - interval1[0]) * (float)(((blockIdx.x*blockDim.x) + (float)threadIdx.x)/totalThreads[0])) + interval1[0])
                    + ((abs(interval2[0] - interval1[0])/totalThreads[0])/2.0f);
            ;


            atomicAdd(&sum, trapezoid((ind - (abs(interval2[0] - interval1[0])/totalThreads[0])/2.0f), (ind + (abs(interval2[0] - interval1[0])/totalThreads[0])/2.0f), f(ind - (abs(interval2[0] - interval1[0])/totalThreads[0])/2.0f), f(ind + (abs(interval2[0] - interval1[0])/totalThreads[0])/2.0f)));
            __syncthreads();

            //Divergence, but only one thread in the block updates sum;
            if (threadIdx.x == 0)
            atomicAdd(output, sum);






}

const float NUM_POLYGONS = 512;
const float INTERVAL1 = 0;
const float INTERVAL2 = 10;
const float NUM_THREADS = 255;
const float NUM_BLOCKS = 1;
const float TOT_THREADS = NUM_THREADS * NUM_BLOCKS;


/**
 * Host main routine
 */
int
main(void)
{


    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("[Trapezoidal Rule]\n");

    // Allocate the variables
    float *numPolygons = (float *)malloc(sizeof(float));
    float *interval1 = (float *)malloc(sizeof(float));
    float *interval2 = (float *)malloc(sizeof(float));
    float *numThreads = (float *)malloc(sizeof(float));
    float *totalThreads = (float *)malloc(sizeof(float));
    float *output = (float *)malloc(sizeof(float));

    if (numPolygons == NULL || interval1 == NULL || interval2 == NULL)
    {
        fprintf(stderr, "Failed to allocate memory!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    numPolygons[0] = NUM_POLYGONS;
    interval1[0] = INTERVAL1;
    interval2[0] = INTERVAL2;
    numThreads[0] = NUM_THREADS;
    totalThreads[0] = TOT_THREADS;
    output[0] = 0;


    // Allocate the device variables
    float *d_numPolygons = NULL;
        err = cudaMalloc((void **)&d_numPolygons, sizeof(float));
    float *d_interval1 = NULL;
        err = cudaMalloc((void **)&d_interval1, sizeof(float));
    float *d_interval2 = NULL;
        err = cudaMalloc((void **)&d_interval2, sizeof(float));
    float *d_numThreads = NULL;
        err = cudaMalloc((void **)&d_numThreads, sizeof(float));
    float *d_totalThreads = NULL;
        err = cudaMalloc((void **)&d_totalThreads, sizeof(float));
    float *d_output = NULL;
        err = cudaMalloc((void **)&d_output, sizeof(float));





    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Copy the host memory to device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_numPolygons, numPolygons, sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_interval1, interval1, sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_interval2, interval2, sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_numThreads, numThreads, sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_totalThreads, totalThreads, sizeof(float), cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_output, output, sizeof(float), cudaMemcpyHostToDevice);



    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy memory from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 threadDim(NUM_THREADS,1,1);
    dim3 blockDim(NUM_BLOCKS,1,1);           //One block

    // Launch the Kernel
    riemannSum<<<blockDim, threadDim>>>(d_interval1, d_interval2, d_totalThreads, d_output);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch trapezoidal sum kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }



    printf("Integral for %f trapezoids on the interval %f to %f is %f \n", NUM_POLYGONS, INTERVAL1, INTERVAL2, output[0]);


    // Free device global memory
    err = cudaFree(d_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device memory \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    // Free host memory


    free(output);

    printf("Done\n");
    return 0;
}
