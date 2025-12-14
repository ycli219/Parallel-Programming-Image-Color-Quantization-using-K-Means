#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define MAX_K 256
#define BLOCK_SIZE 256

// Constant memory for fast broadcast access to centroids
__constant__ float c_centroids[MAX_K * 3];

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)

// Kernel: Assign pixels and perform block-level partial reduction
// Optimization: Uses Shared Memory to avoid global atomic contention
__global__ void kMeansAssignAndPartialReduce(
    const unsigned char* __restrict__ d_img,
    float* d_partial_sums,
    int* d_partial_counts,
    int* d_labels,
    int numPixels,
    int K
) {
    extern __shared__ float s_mem[];
    float* s_sums = s_mem;
    int* s_counts = (int*)&s_sums[K * 3];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared memory
    for (int i = tid; i < K; i += blockDim.x) {
        s_sums[i * 3] = 0.0f; s_sums[i * 3 + 1] = 0.0f; s_sums[i * 3 + 2] = 0.0f;
        s_counts[i] = 0;
    }
    __syncthreads();

    if (gid < numPixels) {
        float b = (float)d_img[gid * 3];
        float g = (float)d_img[gid * 3 + 1];
        float r = (float)d_img[gid * 3 + 2];

        float minDist = 1e20f;
        int bestK = 0;

        // Read centroids from Constant Memory
        for (int k = 0; k < K; ++k) {
            float cb = c_centroids[k * 3];
            float cg = c_centroids[k * 3 + 1];
            float cr = c_centroids[k * 3 + 2];
            float dist = (b - cb)*(b - cb) + (g - cg)*(g - cg) + (r - cr)*(r - cr);
            if (dist < minDist) {
                minDist = dist;
                bestK = k;
            }
        }
        d_labels[gid] = bestK;

        // Accumulate in Shared Memory
        atomicAdd(&s_sums[bestK * 3], b);
        atomicAdd(&s_sums[bestK * 3 + 1], g);
        atomicAdd(&s_sums[bestK * 3 + 2], r);
        atomicAdd(&s_counts[bestK], 1);
    }
    __syncthreads();

    // Write partial block results to Global Memory
    for (int i = tid; i < K; i += blockDim.x) {
        int outIdx = blockIdx.x * K + i;
        d_partial_sums[outIdx * 3]     = s_sums[i * 3];
        d_partial_sums[outIdx * 3 + 1] = s_sums[i * 3 + 1];
        d_partial_sums[outIdx * 3 + 2] = s_sums[i * 3 + 2];
        d_partial_counts[outIdx]       = s_counts[i];
    }
}

// Kernel: Final reduction and centroid update
// Optimization: Runs on GPU to avoid expensive D2H transfers
__global__ void kMeansUpdateCentroids(
    const float* __restrict__ d_partial_sums,
    const int* __restrict__ d_partial_counts,
    float* d_new_centroids,
    float* d_total_diff,
    int numBlocks,
    int K
) {
    __shared__ float s_diff;
    if (threadIdx.x == 0) s_diff = 0.0f;
    __syncthreads();

    int k = threadIdx.x; 
    if (k < K) {
        float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
        int count = 0;

        // Aggregate partial results from all blocks
        for (int b = 0; b < numBlocks; ++b) {
            int idx = b * K + k;
            sumB += d_partial_sums[idx * 3];
            sumG += d_partial_sums[idx * 3 + 1];
            sumR += d_partial_sums[idx * 3 + 2];
            count += d_partial_counts[idx];
        }

        float newB, newG, newR;
        if (count > 0) {
            newB = sumB / count; 
            newG = sumG / count; 
            newR = sumR / count;
        } else {
            // Keep old centroid if cluster is empty
            newB = c_centroids[k * 3]; 
            newG = c_centroids[k * 3 + 1]; 
            newR = c_centroids[k * 3 + 2];
        }

        // Calculate movement diff
        float oldB = c_centroids[k * 3];
        float oldG = c_centroids[k * 3 + 1];
        float oldR = c_centroids[k * 3 + 2];
        float diff = fabsf(newB - oldB) + fabsf(newG - oldG) + fabsf(newR - oldR);

        atomicAdd(&s_diff, diff);

        d_new_centroids[k * 3]     = newB;
        d_new_centroids[k * 3 + 1] = newG;
        d_new_centroids[k * 3 + 2] = newR;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *d_total_diff = s_diff;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) return -1;
    std::string inputPath = argv[1];
    std::string outputPath = argv[2];
    int K = std::atoi(argv[3]);

    if (K <= 0 || K > MAX_K) {
        std::cerr << "Error: K must be between 1 and " << MAX_K << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(inputPath);
    if (img.empty()) return -1;
    if (!img.isContinuous()) img = img.clone();

    int numPixels = img.cols * img.rows;
    size_t imgSize = numPixels * 3 * sizeof(unsigned char);

    std::srand(std::time(0)); 
    std::vector<float> h_centroids(K * 3);
    for (int i = 0; i < K; ++i) {
        int idx = std::rand() % numPixels;
        h_centroids[i * 3]     = img.data[idx * 3];
        h_centroids[i * 3 + 1] = img.data[idx * 3 + 1];
        h_centroids[i * 3 + 2] = img.data[idx * 3 + 2];
    }

    unsigned char *d_img;
    int *d_labels;
    float *d_partial_sums, *d_centroids_global, *d_diff;
    int *d_partial_counts;

    int gridSize = (numPixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

    CUDA_CHECK(cudaMalloc(&d_img, imgSize));
    CUDA_CHECK(cudaMalloc(&d_labels, numPixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partial_sums, gridSize * K * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_partial_counts, gridSize * K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_centroids_global, K * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diff, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_img, img.data, imgSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids_global, h_centroids.data(), K * 3 * sizeof(float), cudaMemcpyHostToDevice));

    size_t smemSizeK1 = K * 3 * sizeof(float) + K * sizeof(int);

    for (int iter = 0; iter < 50; ++iter) {
        // Update Constant Memory directly from Device Memory
        CUDA_CHECK(cudaMemcpyToSymbol(c_centroids, d_centroids_global, K * 3 * sizeof(float), 0, cudaMemcpyDeviceToDevice));
        
        // Kernel 1: Assignment and Partial Reduction
        kMeansAssignAndPartialReduce<<<gridSize, BLOCK_SIZE, smemSizeK1>>>(
            d_img, d_partial_sums, d_partial_counts, d_labels, numPixels, K
        );
        CUDA_CHECK(cudaGetLastError());

        // Kernel 2: Final Reduction and Centroid Update
        int updateThreads = (K < 32) ? 32 : K; 
        kMeansUpdateCentroids<<<1, updateThreads>>>(
            d_partial_sums, d_partial_counts, d_centroids_global, d_diff, gridSize, K
        );
        CUDA_CHECK(cudaGetLastError());

        // Check Convergence
        float totalMove = 0.0f;
        CUDA_CHECK(cudaMemcpy(&totalMove, d_diff, sizeof(float), cudaMemcpyDeviceToHost));

        if (totalMove < 1.0f) break;
    }


    std::vector<int> final_labels(numPixels);
    CUDA_CHECK(cudaMemcpy(final_labels.data(), d_labels, numPixels * sizeof(int), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(h_centroids.data(), d_centroids_global, K * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    cv::Mat outputImg = img.clone();
    for (int i = 0; i < numPixels; ++i) {
        int id = final_labels[i];
        outputImg.data[i * 3]     = (unsigned char)h_centroids[id * 3];
        outputImg.data[i * 3 + 1] = (unsigned char)h_centroids[id * 3 + 1];
        outputImg.data[i * 3 + 2] = (unsigned char)h_centroids[id * 3 + 2];
    }
    cv::imwrite(outputPath, outputImg);

    cudaFree(d_img); cudaFree(d_labels);
    cudaFree(d_partial_sums); cudaFree(d_partial_counts);
    cudaFree(d_centroids_global); cudaFree(d_diff);

    return 0;
}