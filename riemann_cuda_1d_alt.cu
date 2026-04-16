#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// --- KERNEL CUDA ---
// Menghitung kontribusi luas setiap sub-interval secara paralel
__global__ void riemannSumKernel(float a, float dx, int n, double* d_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n) {
        // 1. Hitung titik tengah (xi) dari sub-interval ke-i
        // xi = a + (i + 0.5) * delta_x
        float xi = a + (i + 0.5f) * dx;
        
        // 2. Hitung f(xi) * delta_x, di mana f(x) = x^2
        double area = (double)(xi * xi) * dx;
        
        // 3. Tambahkan ke total akumulasi di memori global
        // Menggunakan atomicAdd agar tidak terjadi race condition antar thread
        atomicAdd(d_sum, area);
    }
}

void solveRiemann(int n) {
    float a = -1.0f;
    float b = 2.0f;
    float dx = (b - a) / n;
    double h_sum = 0.0;
    double* d_sum;

    // --- PENGUKURAN WAKTU E2E (End-to-End) ---
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Alokasi memori di GPU
    cudaMalloc(&d_sum, sizeof(double));
    
    // 2. Inisialisasi nilai di GPU (set ke 0)
    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);

    // 3. Konfigurasi Grid dan Block
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Launch Kernel
    riemannSumKernel<<<blocksPerGrid, threadsPerBlock>>>(a, dx, n, d_sum);
    
    // Sinkronisasi untuk memastikan kernel selesai sebelum memori ditarik
    cudaDeviceSynchronize();

    // 5. Salin hasil kembali ke Host (CPU)
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

    // 6. Bebaskan memori GPU
    cudaFree(d_sum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // --- KALKULASI ERROR ---
    double analyticalValue = 3.0;
    double error = std::abs(h_sum - analyticalValue);

    // Output Hasil
    std::cout << "N: " << n 
              << " | Hasil: " << h_sum 
              << " | Error: " << error 
              << " | Waktu E2E: " << elapsed.count() << " ms" << std::endl;
}

int main() {
    std::vector<int> partitions = {100, 1000, 10000, 100000};

    std::cout << "Menghitung Integral x^2 dari [-1, 2] dengan CUDA\n";
    std::cout << "-----------------------------------------------\n";

    for (int n : partitions) {
        solveRiemann(n);
    }

    return 0;
}