#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>

// Nilai eksak integral (x^2 + y^2) pada domain [-1,2]x[-1,2] adalah 18.0
double analyticalValue = 18.0;

// Fungsi f(x,y) = x^2 + y^2
double f(double x, double y) {
    return (x * x) + (y * y);
}

void solveRiemann2D(long long n) {
    // Domain [-1, 2] untuk x dan y
    double a = -1.0, b = 2.0; // Rentang x
    double c = -1.0, d = 2.0; // Rentang y
    
    double dx = (b - a) / n;
    double dy = (d - c) / n;
    double total_volume = 0.0;

    // Ukur waktu start
    auto start = std::chrono::high_resolution_clock::now();

    // Loop nested untuk 2 Dimensi (n x n partisi)
    for (long long i = 0; i < n; i++) {
        double x = a + (i + 0.5) * dx; // Titik tengah x
        for (long long j = 0; j < n; j++) {
            double y = c + (j + 0.5) * dy; // Titik tengah y
            
            // Volume setiap "tiang" = f(x,y) * luas alas (dx * dy)
            total_volume += f(x, y) * dx * dy;
        }
    }

    // Ukur waktu end 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    // Kalkulasi error
    double error = std::abs(total_volume - analyticalValue);

    // Output Hasil
    std::cout << "N: " << n << "x" << n 
              << " | Total Partisi: " << n * n
              << " | Hasil: " << total_volume 
              << " | Error: " << error 
              << " | Waktu (ms): " << elapsed.count() << " ms" << std::endl;
}

int main() {
    long long option;
    std::cout << "run program langsung? 1 = ya : ";
    std::cin >> option;

    std::cout << "Menghitung Integral 2D (x^2 + y^2) dari [-1, 2] secara serial\n";
    std::cout << "----------------------------------------------------------\n";

    if (option == 1) {
        // Daftar N (partisi per sisi). Total partisi = N * N
        std::vector<long long> partitions = {
            1000,       // Total 1.000.000
            10'000,     // Total 100.000.000
            30'000,     // Total 900.000.000 (Mulai terasa berat di serial)
            100'000, // Total 10.000.000.000 (Sangat lama di serial)
        };
        for (long long n : partitions) {
            solveRiemann2D(n);
        }
    } else {
        long long n;
        std::cout << "Masukkan N (partisi per sisi): ";
        std::cin >> n;
        solveRiemann2D(n);
    }

    return 0;
}