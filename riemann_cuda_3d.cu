#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// --- KERNEL CUDA 3D ---
// Menghitung kontribusi nilai fungsi di setiap voxel (volume pixel)
__global__ void riemannSum3DKernel(float a, float c, float e, float dx, float dy, float dz, 
                                   int nx, int ny, int nz, double* d_sum) {
    // Menghitung indeks global 3D: i (x), j (y), k (z)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Pastikan indeks berada dalam rentang partisi
    if (i < nx && j < ny && k < nz) {
        // 1. Cari titik tengah voxel (xi, yj, zk)
        float xi = a + (i + 0.5f) * dx;
        float yj = c + (j + 0.5f) * dy;
        float zk = e + (k + 0.5f) * dz;

        // 2. Evaluasi fungsi f(x,y,z) = x^2 + y^2 + z^2
        float f_xyz = (xi * xi) + (yj * yj) + (zk * zk);

        // 3. Hitung kontribusi volume: f(x,y,z) * delta_V
        double voxelContribution = (double)f_xyz * dx * dy * dz;

        // 4. Akumulasi secara atomik ke memori global
        atomicAdd(d_sum, voxelContribution);
    }
}

void runSimulation3D(int n) {
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
    std::vector<int> partitions = {100, 500, 1000};

    std::cout << "Menghitung Integral 3D (x^2 + y^2 + z^2) dengan CUDA\n";
    std::cout << "--------------------------------------------------\n";

    for (int n : partitions) {
        runSimulation3D(n);
    }

    return 0;
}