#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// --- KERNEL CUDA 2D ---
// Menghitung volume setiap "tiang" kecil secara paralel dalam grid 2D
__global__ void riemannSum2DKernel(float a, float c, float dx, float dy, int nx, int ny, double* d_sum) {
    // Menghitung indeks global i (x) dan j (y)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        // 1. Hitung titik tengah (xi*, yj*) untuk sel (i, j)
        float xi = a + (i + 0.5f) * dx;
        float yj = c + (j + 0.5f) * dy;

        // 2. Hitung f(xi, yj) = x^2 + y^2
        float f_xy = (xi * xi) + (yj * yj);

        // 3. Volume sel = f(xi, yj) * delta_A (di mana delta_A = dx * dy)
        double cellVolume = (double)f_xy * dx * dy;

        // 4. Akumulasi ke memori global secara atomik
        atomicAdd(d_sum, cellVolume);
    }
}

void solveRiemann(int n) {
    // Domain [-1, 2] untuk x dan y
    float a = -1.0f, b = 2.0f;
    float c = -1.0f, d = 2.0f;
    
    int nx = n, ny = n; // Menggunakan partisi n x n
    float dx = (b - a) / nx;
    float dy = (d - c) / ny;
    
    double h_sum = 0.0;
    double* d_sum;

    auto start = std::chrono::high_resolution_clock::now();

    // Persiapan Memori GPU
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);

    // Konfigurasi Grid 2D
    // Menggunakan block 16x16 threads (total 256 thread per block)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch Kernel 2D
    riemannSum2DKernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, dx, dy, nx, ny, d_sum);
    
    cudaDeviceSynchronize();

    // Salin hasil kembali
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Nilai eksak integral (x^2 + y^2) pada [-1,2]x[-1,2] adalah 18.0
    double analyticalValue = 18.0;
    double error = std::abs(h_sum - analyticalValue);

    std::cout << "Partisi: " << nx << "x" << ny 
              << " | Hasil: " << h_sum 
              << " | Error: " << error 
              << " | Waktu E2E: " << elapsed.count() << " ms" << std::endl;
}

int main() {
    // Karena 2D, total thread adalah N^2 (1000x1000 = 1 juta thread)
    std::vector<int> partitions = {100, 1000, 10000};

    std::cout << "Menghitung Integral 2D (x^2 + y^2) dengan CUDA\n";
    std::cout << "----------------------------------------------\n";

    for (int n : partitions) {
        solveRiemann(n);
    }

    return 0;
}