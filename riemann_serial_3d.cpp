#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// Nilai eksak integral (x^2 + y^2 + z^2) pada domain [-1,2]^3 adalah 81.0
double analyticalValue = 81.0;

// Fungsi f(x,y,z) = x^2 + y^2 + z^2
double f(double x, double y, double z) {
    return (x * x) + (y * y) + (z * z);
}

void solveRiemann3D(long long n) {
    // Batas interval [-1, 2] untuk x, y, dan z
    double a = -1.0, b = 2.0; // x
    double c = -1.0, d = 2.0; // y
    double e = -1.0, g = 2.0; // z
    
    double dx = (b - a) / n;
    double dy = (d - c) / n;
    double dz = (g - e) / n;
    double total_sum = 0.0;

    // Ukur waktu start
    auto start = std::chrono::high_resolution_clock::now();

    // Triple nested loop untuk 3 Dimensi (n x n x n partisi)
    for (long long i = 0; i < n; i++) {
        double x = a + (i + 0.5) * dx;
        for (long long j = 0; j < n; j++) {
            double y = c + (j + 0.5) * dy;
            for (long long k = 0; k < n; k++) {
                double z = e + (k + 0.5) * dz;
                
                // Kontribusi = f(x,y,z) * dV (di mana dV = dx * dy * dz)
                total_sum += f(x, y, z) * dx * dy * dz;
            }
        }
    }

    // Ukur waktu end 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Kalkulasi error
    double error = std::abs(total_sum - analyticalValue);

    // Output Hasil
    std::cout << "N: " << n << "x" << n << "x" << n 
              << " | Total Partisi: " << n * n * n
              << " | Hasil: " << total_sum 
              << " | Error: " << error 
              << " | Waktu (ms): " << elapsed.count() << " ms" << std::endl;
}

int main() {
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