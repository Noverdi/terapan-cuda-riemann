#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

// --- KERNEL CUDA ---
// Menghitung kontribusi luas setiap sub-interval secara paralel
__global__ void riemannSumKernel(
    double a, double dx, long long n, double* d_sum) {
    __shared__ double cache[256];

    long long tid = threadIdx.x;
    long long i = (long long)blockIdx.x * blockDim.x + tid;
    
    double area = 0;
    if (i < n) {
        // 1. Hitung titik tengah (xi) dari sub-interval ke-i
        // dengan kata lain adalah menentukan letaknya di koordinat x
        // i * dx menggeser posisi titik a sejauh i * dx
        // xi = a + (i + 0.5) * delta_x
        double xi = a + (i + 0.5) * dx;
        
        // 2. Hitung f(xi) * delta_x, di mana f(x) = x^2
        double f_x = xi * xi;
        area = f_x * dx;
   }
    cache[tid] = area; // simpan hasil dari masing-masing threads ke cache block

    // sinkronisasi, memastikan semua thread di block ini selesai menulis ke cache
    __syncthreads(); 
    
    // 3. reduction (perjumlahan secara bertahap ke dalam block)
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        cache[tid] += cache[tid + s];
    }
    __syncthreads(); //tunggu setiap level penjumlahan selesai
    }

    // 4. hanya thread 0 dari setiap block yang melapor ke memori global
    if (tid == 0) {
    atomicAdd(d_sum, cache[0]);
    }
 
}

void solveRiemann(long long n) {
    double a = -1.0;
    double b = 2.0;

    double dx = (b - a) / n;
    double h_sum = 0.0;
    double* d_sum;

    // --- PENGUKURAN WAKTU E2E (End-to-End) ---
    auto start = std::chrono::high_resolution_clock::now();

    // 1. Alokasi memori di GPU
    cudaMalloc(&d_sum, sizeof(double));
    
    // 2. Inisialisasi nilai di GPU (set ke 0)
    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);

    // 3. Konfigurasi Grid dan Block
    long threadsPerBlock = 256;
    long blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

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
    long long option;
    // Read integer (stops at space)
    std::cout << "run program langsung? 1 = ya : ";
    std::cin >> option;

    std::cout << "Menghitung Integral x^2 dari [-1, 2] dengan CUDA\n";
    std::cout << "-----------------------------------------------\n";

    if (option==1) {
        std::vector<long long> partitions = {
            100'000,
            1000'000,
            1000'000'000,
            10'000'000'000LL,
            100'000'000'000LL,
        };
        for (long long n : partitions) {
            solveRiemann(n);
        }
    } else {
        long long n;
        std::cout << "Masukkan N : ";
        std::cin >> n;
        solveRiemann(n);
    }

    return 0;
}