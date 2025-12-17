#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <mpi.h>
#include <opencv2/opencv.hpp>

// Structure to simplify centroid handling in flat arrays
struct Centroid {
    double b, g, r;
};

// Helper function to calculate squared Euclidean distance
// Using squared distance avoids costly sqrt() without changing the comparison result
double getDistanceSq(uchar b1, uchar g1, uchar r1, double b2, double g2, double r2) {
    return std::pow(b1 - b2, 2) + std::pow(g1 - g2, 2) + std::pow(r1 - r2, 2);
}

int main(int argc, char** argv) {
    // 1. MPI Initialization
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Variables for Rank 0
    std::string imgFileName, outFileName;
    int K = 64;
    cv::Mat originalImage;
    int total_pixels = 0;
    int rows = 0, cols = 0;
    int channels = 3;

    // 2. Argument Parsing and Image Loading (Rank 0 only)
    if (world_rank == 0) {
        if (argc < 3) {
            std::cerr << "Usage: mpirun -np <N> ./prog <input> <output> [K]\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        imgFileName = argv[1];
        outFileName = argv[2];
        if (argc == 4) K = std::stoi(argv[3]);

        originalImage = cv::imread(imgFileName);
        if (originalImage.empty()) {
            std::cerr << "Error: Could not open image.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        rows = originalImage.rows;
        cols = originalImage.cols;
        total_pixels = rows * cols;
    }

    // 3. Broadcast Metadata
    // Everyone needs to know K and the total data size to prepare buffers
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_pixels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 4. Calculate Data Distribution (Scatterv Setup)
    // We need to split total_pixels among world_size processes.
    // Since it might not divide evenly, we use counts and displs arrays.
    std::vector<int> send_counts(world_size);
    std::vector<int> displs(world_size);

    int remainder = total_pixels % world_size;
    int sum = 0;

    // Calculate how many PIXELS each rank gets
    for (int i = 0; i < world_size; i++) {
        send_counts[i] = total_pixels / world_size;
        if (i < remainder) send_counts[i]++;
        displs[i] = sum;
        sum += send_counts[i];
    }

    // Convert PIXEL counts to BYTE counts (x3 for BGR) for MPI communication
    std::vector<int> send_bytes(world_size);
    std::vector<int> displs_bytes(world_size);
    for(int i=0; i<world_size; i++) {
        send_bytes[i] = send_counts[i] * channels;
        displs_bytes[i] = displs[i] * channels;
    }

    // Allocate local buffer for this process
    int local_n = send_counts[world_rank];
    std::vector<uchar> local_image_data(local_n * channels);

    // Prepare input buffer (only relevant on Rank 0)
    uchar* input_data_ptr = nullptr;
    if (world_rank == 0) {
        // Flattened access to OpenCV Mat data
        input_data_ptr = originalImage.ptr<uchar>(0);
    }

    // 5. Scatter Image Data
    MPI_Scatterv(input_data_ptr, send_bytes.data(), displs_bytes.data(), MPI_UNSIGNED_CHAR,
                 local_image_data.data(), local_n * channels, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    // 6. Centroid Initialization
    // We use a flat vector for centroids: [b0, g0, r0, b1, g1, r1, ...]
    std::vector<double> centroids(K * 3);

    if (world_rank == 0) {
        std::cout << "Broadcasting initial centroids...\n";
        // Initialize randomly using same logic as sequential
        std::default_random_engine dre(std::chrono::steady_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<int> uid(0, total_pixels - 1);

        for (int i = 0; i < K; i++) {
            int randIdx = uid(dre);
            // Access pixel at random index
            // Note: OpenCV data is contiguous BGR
            uchar b = input_data_ptr[randIdx * 3 + 0];
            uchar g = input_data_ptr[randIdx * 3 + 1];
            uchar r = input_data_ptr[randIdx * 3 + 2];
            centroids[i * 3 + 0] = (double)b;
            centroids[i * 3 + 1] = (double)g;
            centroids[i * 3 + 2] = (double)r;
        }
    }

    // Broadcast initial centroids to all ranks
    MPI_Bcast(centroids.data(), K * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 7. Training Loop
    int iterations = 50; 
    for (int iter = 0; iter < iterations; iter++) {
        if (world_rank == 0) std::cout << "Iteration " << iter + 1 << "/" << iterations << "...\n";

        // Structures for Local Accumulation
        // local_sums: Sum of [B, G, R] for each cluster K
        std::vector<double> local_sums(K * 3, 0.0);
        std::vector<int> local_counts(K, 0);

        // --- Step A: Assignment (Local Parallel) ---
        // Iterate over local pixels
        for (int i = 0; i < local_n; i++) {
            uchar b = local_image_data[i * 3 + 0];
            uchar g = local_image_data[i * 3 + 1];
            uchar r = local_image_data[i * 3 + 2];

            int bestCluster = 0;
            double minDist = -1.0;

            // Find nearest centroid
            for (int k = 0; k < K; k++) {
                double dist = getDistanceSq(b, g, r, 
                                            centroids[k*3+0], 
                                            centroids[k*3+1], 
                                            centroids[k*3+2]);
                if (minDist < 0 || dist < minDist) {
                    minDist = dist;
                    bestCluster = k;
                }
            }

            // Accumulate locally
            local_sums[bestCluster * 3 + 0] += b;
            local_sums[bestCluster * 3 + 1] += g;
            local_sums[bestCluster * 3 + 2] += r;
            local_counts[bestCluster]++;
        }

        // --- Step B: Update (Global Reduction) ---
        std::vector<double> global_sums(K * 3);
        std::vector<int> global_counts(K);

        // Reduce sums and counts from all processes to all processes
        MPI_Allreduce(local_sums.data(), global_sums.data(), K * 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // Update Centroids
        for (int k = 0; k < K; k++) {
            if (global_counts[k] > 0) {
                centroids[k * 3 + 0] = global_sums[k * 3 + 0] / global_counts[k];
                centroids[k * 3 + 1] = global_sums[k * 3 + 1] / global_counts[k];
                centroids[k * 3 + 2] = global_sums[k * 3 + 2] / global_counts[k];
            }
            // If a cluster is empty, it keeps its old position (or could be re-initialized)
        }
    }

    // 8. Reconstruction (Convert)
    // Update local pixels to their final centroid color
    for (int i = 0; i < local_n; i++) {
        uchar b = local_image_data[i * 3 + 0];
        uchar g = local_image_data[i * 3 + 1];
        uchar r = local_image_data[i * 3 + 2];

        int bestCluster = 0;
        double minDist = -1.0;

        for (int k = 0; k < K; k++) {
            double dist = getDistanceSq(b, g, r, 
                                        centroids[k*3+0], 
                                        centroids[k*3+1], 
                                        centroids[k*3+2]);
            if (minDist < 0 || dist < minDist) {
                minDist = dist;
                bestCluster = k;
            }
        }

        // Replace pixel color with centroid color
        local_image_data[i * 3 + 0] = (uchar)centroids[bestCluster * 3 + 0];
        local_image_data[i * 3 + 1] = (uchar)centroids[bestCluster * 3 + 1];
        local_image_data[i * 3 + 2] = (uchar)centroids[bestCluster * 3 + 2];
    }

    // 9. Gather Data (Gatherv)
    // Create buffer for final image on Rank 0
    std::vector<uchar> final_image_data;
    if (world_rank == 0) {
        final_image_data.resize(total_pixels * channels);
    }

    MPI_Gatherv(local_image_data.data(), local_n * channels, MPI_UNSIGNED_CHAR,
                final_image_data.data(), send_bytes.data(), displs_bytes.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // 10. Save Output (Rank 0 only)
    if (world_rank == 0) {
        std::cout << "Reconstructing image...\n";
        // Copy data back to Mat
        // Because vector is contiguous and Mat is contiguous (usually), memcpy works
        // Or simply assign if strict pointer safety is needed
        std::memcpy(originalImage.ptr(), final_image_data.data(), final_image_data.size());

        cv::imwrite(outFileName, originalImage);
        std::cout << "Done. Saved to " << outFileName << "\n";
    }

    MPI_Finalize();
    return 0;
}