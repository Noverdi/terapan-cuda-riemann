#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// --- KERNEL CUDA 2D ---
// Menghitung volume setiap "tiang" kecil secara paralel dalam grid 2D
__global__ void riemannSum2DKernel(
    double a, double c, double dx, double dy, long long nx, long long ny, double* d_sum) {
    __shared__ double cache[256] ; // 16 x 16

    // Menghitung indeks global i (x) dan j (y)
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long j = (long long)blockIdx.y * blockDim.y + threadIdx.y;

    // indeks linear untuk shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    double cellVolume = 0.0;
    if (i < nx && j < ny) {
        // 1. Hitung titik tengah (xi*, yj*) untuk sel (i, j)
        double xi = a + (i + 0.5) * dx;
        double yj = c + (j + 0.5) * dy;

        // 2. Hitung f(xi, yj) = x^2 + y^2
        double f_xy = (xi * xi) + (yj * yj);

        // 3. Volume sel = f(xi, yj) * delta_A (di mana delta_A = dx * dy)
        cellVolume = f_xy * dx * dy;

        // 4. Akumulasi ke memori global secara atomik
        // atomicAdd(d_sum, cellVolume);
    }
    //simpan ke cache
    cache[tid] = cellVolume;
    __syncthreads();

    //reduction di dalam bloick (jumlah)
    for (int s = (blockDim.x * blockDim.y)/2; s > 0; s >>=1) {
        if (tid<s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }
    // satu thread per block yang nulis ke memori global
    if (tid==0){
        atomicAdd(d_sum,cache[0]);
    }
}

void solveRiemann(long long n) {
    // Domain [-1, 2] untuk x dan y
    double a = -1.0, b = 2.0;
    double c = -1.0, d = 2.0;
    
    long long nx = n, ny = n; // Menggunakan partisi n x n
    double dx = (b - a) / nx;
    double dy = (d - c) / ny;
    
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
    long long option;
    // Read long longeger (stops at space)
    std::cout << "run program langsung? 1 = ya : ";
    std::cin >> option;

    std::cout << "Menghitung integral (x^2 + y^2) pada [-1,2]x[-1,2] menggunakan CUDA\n";
    std::cout << "-----------------------------------------------\n";

    if (option==1) {
        std::vector<long long> partitions = {  // karena 2 dimensi, 1000x1000 = sejuta partisi
            1000,       // Total 1.000.000
            10'000,     // Total 100.000.000
            30'000,     // Total 900.000.000 (Mulai terasa berat di serial)
            100'000,    // Total 10.000.000.000 (Sangat lama di serial)
            1000'000,
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