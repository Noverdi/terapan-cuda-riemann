#include <iostream>
#include <chrono>
#include <vector>

// fungsi yang digunakan adalah f(x) = x^2 dengan batas bawah -1 dan batas atas 2
// maka hasil integralnya adalah 3, akan dibuat terlebih dahulu 
// variabel hasil sebagai standar perhitungan error

double analyticalValue = 3.0f;

// fungsi f(x) = x^2
double f(double x) {
    return x * x;
}

void solveRiemann(long long n) {
    double a = -1.0; // batas bawah
    double b = 2.0; // batas atas
    double dx = (b-a)/n; // lebar persegi panjang di x
    double total_area = 0.0;

    // ukur waktu start
    auto start = std::chrono::high_resolution_clock::now();

    for (long long i = 0; i<n; i++) {
        double x = a + (i + 0.5) * dx; // itung posisi x, a + pergeseran di sumbu x
        total_area += f(x) * dx; // perhitungan total area persegi panjang per partisi
    }
    // ukur waktu end 
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed =  end-start;

    // kalkulasi error
    double error = std::abs(total_area - analyticalValue);

    // Output Hasil
    std::cout << "N: " << n 
              << " | Hasil: " << total_area 
              << " | Error: " << error 
              << " | Waktu (ms): " << elapsed.count() << " ms" << std::endl;

}


int main() {
    long long option;
    // Read integer (stops at space)
    std::cout << "run program langsung? 1 = ya : ";
    std::cin >> option;

    std::cout << "Menghitung Integral x^2 dari [-1, 2] secara serial\n";
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