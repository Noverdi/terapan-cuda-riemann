#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// --- KERNEL CUDA 3D ---
// Menghitung kontribusi nilai fungsi di setiap voxel (volume pixel)
__global__ void riemannSum3DKernel(double a, double c, double e, double dx, double dy, double dz, 
                                   long long nx, long long ny, long long nz, double* d_sum) {
    // 1. Deklarasi Cache (8*8*4 = 256 thread)
    __shared__ double cache[256];

    // Indeks global (Casting ke long long agar tidak overflow)
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long j = (long long)blockIdx.y * blockDim.y + threadIdx.y;
    long long k = (long long)blockIdx.z * blockDim.z + threadIdx.z;

    // Indeks linear untuk akses shared memory (0 - 255)
    int tid = (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    double voxelValue = 0.0;
    if (i < nx && j < ny && k < nz) {
        double xi = a + (i + 0.5) * dx;
        double yj = c + (j + 0.5) * dy;
        double zk = e + (k + 0.5) * dz;

        double f_xyz = (xi * xi) + (yj * yj) + (zk * zk);
        voxelValue = f_xyz * (dx * dy * dz);
    }

    // Simpan ke cache dan sinkronisasi
    cache[tid] = voxelValue;
    __syncthreads();

    // 2. Parallel Reduction (Pohon Penjumlahan)
    for (int s = (blockDim.x * blockDim.y * blockDim.z) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();
    }

    // 3. Hanya satu thread per block yang menulis ke Global Memory
    if (tid == 0) {
        atomicAdd(d_sum, cache[0]);
    }
}

void solveRiemann3D(int n) {
    // Batas interval [-1, 2] untuk ketiga dimensi
    float a = -1.0f, b = 2.0f; // x
    float c = -1.0f, d = 2.0f; // y
    float e = -1.0f, f = 2.0f; // z
    
    int nx = n, ny = n, nz = n; 
    float dx = (b - a) / nx;
    float dy = (d - c) / ny;
    float dz = (f - e) / nz;
    
    double h_sum = 0.0;
    double* d_sum;

    auto start = std::chrono::high_resolution_clock::now();

    // Alokasi dan Inisialisasi GPU
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);

    // Konfigurasi Grid & Block 3D
    // Menggunakan block 8x8x4 (total 256 thread) agar pas dengan arsitektur GPU modern
    dim3 threadsPerBlock(8, 8, 4);
    dim3 blocksPerGrid((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Eksekusi Kernel
    riemannSum3DKernel<<<blocksPerGrid, threadsPerBlock>>>(a, c, e, dx, dy, dz, nx, ny, nz, d_sum);
    
    cudaDeviceSynchronize();

    // Ambil hasil dan bebaskan memori
    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Nilai analitik: 81.0
    double analyticalValue = 81.0;
    double error = std::abs(h_sum - analyticalValue);

    std::cout << "Partisi: " << n << "^3 (" << (long)n*n*n << " voxel)" 
              << " | Hasil: " << h_sum 
              << " | Error: " << error 
              << " | Waktu E2E: " << elapsed.count() << " ms" << std::endl;
}

int main() {
    // Untuk 3D, n=200 saja sudah berarti 8 juta thread.
    // n=500 berarti 125 juta thread. Hati-hati dengan timeout kernel pada n yang sangat besar.
    long long option;
    std::cout << "run program langsung? 1 = ya : ";
    std::cin >> option;

    std::cout << "Menghitung Integral 3D (x^2 + y^2 + z^2) dari [-1, 2]^3 secara serial\n";
    std::cout << "------------------------------------------------------------------\n";

    if (option == 1) {
        // Daftar N (partisi per sisi). Total partisi = N^3
        std::vector<long long> partitions = {
            1000,    // Total 1.000.000.000 (Mulai terasa berat di serial)
            2000,    // Total 8.000.000.000 (Akan sangat lama di serial)
            3000,    // Total 27.000.000.000 (Akan sangat lama di serial)
            4000,    // Total 64.000.000.000 (Akan sangat lama di serial)
        };
        for (long long n : partitions) {
            solveRiemann3D(n);
        }
    } else {
        long long n;
        std::cout << "Masukkan N (partisi per sisi): ";
        std::cin >> n;
        solveRiemann3D(n);
    }

    return 0;
}