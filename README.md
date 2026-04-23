# Implementasi CUDA untuk Integral Riemann

Project ini berisi implementasi metode integral Riemann midpoint secara serial dan paralel menggunakan CUDA. Program menghitung integral 1D, 2D, dan 3D, lalu membandingkan akurasi serta waktu eksekusi antara CPU serial dan GPU CUDA.

Data performa pada README ini dirangkum dari file benchmark lokal `rekap_integral_lengkap.csv`. File CSV tersebut diabaikan oleh `.gitignore`, sehingga hasil pentingnya sudah disalin langsung ke tabel pada README.

## Ringkasan Metode

Metode yang digunakan adalah Riemann midpoint, yaitu mengambil titik tengah setiap partisi sebagai sampel fungsi.

| Dimensi | Fungsi | Domain | Nilai analitik |
|---|---|---|---:|
| 1D | `f(x) = x^2` | `[-1, 2]` | `3` |
| 2D | `f(x, y) = x^2 + y^2` | `[-1, 2] x [-1, 2]` | `18` |
| 3D | `f(x, y, z) = x^2 + y^2 + z^2` | `[-1, 2]^3` | `81` |

## Formulasi Integral Riemann

Implementasi pada project ini memakai pendekatan titik tengah atau midpoint rule. Setiap interval dibagi menjadi beberapa partisi kecil, lalu nilai fungsi dihitung pada titik tengah partisi tersebut.

### 1D

Untuk integral satu dimensi:

```math
\int_a^b f(x)\,dx \approx \sum_{i=0}^{N-1} f(x_i^*) \Delta x
```

dengan:

```math
\Delta x = \frac{b-a}{N}
```

```math
x_i^* = a + \left(i + \frac{1}{2}\right)\Delta x
```

Pada project ini:

```math
\int_{-1}^{2} x^2\,dx \approx \sum_{i=0}^{N-1} (x_i^*)^2 \Delta x = 3
```

### 2D

Untuk integral dua dimensi:

```math
\int_a^b \int_c^d f(x,y)\,dy\,dx \approx
\sum_{i=0}^{N_x-1}\sum_{j=0}^{N_y-1} f(x_i^*, y_j^*) \Delta x \Delta y
```

dengan:

```math
\Delta x = \frac{b-a}{N_x}, \quad
\Delta y = \frac{d-c}{N_y}
```

```math
x_i^* = a + \left(i + \frac{1}{2}\right)\Delta x, \quad
y_j^* = c + \left(j + \frac{1}{2}\right)\Delta y
```

Pada project ini:

```math
\int_{-1}^{2}\int_{-1}^{2} (x^2 + y^2)\,dy\,dx \approx
\sum_{i=0}^{N-1}\sum_{j=0}^{N-1} ((x_i^*)^2 + (y_j^*)^2)\Delta x \Delta y = 18
```

### 3D

Untuk integral tiga dimensi:

```math
\int_a^b \int_c^d \int_e^g f(x,y,z)\,dz\,dy\,dx \approx
\sum_{i=0}^{N_x-1}\sum_{j=0}^{N_y-1}\sum_{k=0}^{N_z-1}
f(x_i^*, y_j^*, z_k^*) \Delta x \Delta y \Delta z
```

dengan:

```math
\Delta x = \frac{b-a}{N_x}, \quad
\Delta y = \frac{d-c}{N_y}, \quad
\Delta z = \frac{g-e}{N_z}
```

```math
x_i^* = a + \left(i + \frac{1}{2}\right)\Delta x, \quad
y_j^* = c + \left(j + \frac{1}{2}\right)\Delta y, \quad
z_k^* = e + \left(k + \frac{1}{2}\right)\Delta z
```

Pada project ini:

```math
\int_{-1}^{2}\int_{-1}^{2}\int_{-1}^{2} (x^2 + y^2 + z^2)\,dz\,dy\,dx \approx
\sum_{i=0}^{N-1}\sum_{j=0}^{N-1}\sum_{k=0}^{N-1}
((x_i^*)^2 + (y_j^*)^2 + (z_k^*)^2)\Delta x \Delta y \Delta z = 81
```

Pada versi CUDA, setiap thread menghitung kontribusi satu partisi atau sel, lalu hasilnya dijumlahkan menggunakan shared memory reduction di dalam block dan `atomicAdd` ke memori global.


## Struktur File Repository

| File | Keterangan |
|---|---|
| `riemann_serial_1d.cpp` | Implementasi integral Riemann 1D secara serial |
| `riemann_serial_2d.cpp` | Implementasi integral Riemann 2D secara serial |
| `riemann_serial_3d.cpp` | Implementasi integral Riemann 3D secara serial |
| `riemann_cuda_1d_alt.cu` | Implementasi integral Riemann 1D dengan CUDA |
| `riemann_cuda_2d.cu` | Implementasi integral Riemann 2D dengan CUDA |
| `riemann_cuda_3d.cu` | Implementasi integral Riemann 3D dengan CUDA |
| `README.md` | Dokumentasi project dan ringkasan hasil benchmark |

## Kebutuhan

- Compiler C++ seperti `g++`
- NVIDIA CUDA Toolkit, termasuk `nvcc`
- GPU NVIDIA yang mendukung CUDA

## Cara Compile

Compile program serial:

```bash
g++ -O2 riemann_serial_1d.cpp -o riemann_serial
g++ -O2 riemann_serial_2d.cpp -o riemann_serial_2d
g++ -O2 riemann_serial_3d.cpp -o riemann_serial_3d
```

Compile program CUDA:

```bash
nvcc -O2 riemann_cuda_1d_alt.cu -o riemann_1d
nvcc -O2 riemann_cuda_2d.cu -o riemann_2d
nvcc -O2 riemann_cuda_3d.cu -o riemann_3d
```

## Cara Menjalankan

Jalankan salah satu binary, lalu masukkan `1` untuk menjalankan daftar partisi bawaan:

```bash
./riemann_serial
./riemann_serial_2d
./riemann_serial_3d
./riemann_1d
./riemann_2d
./riemann_3d
```

## Hasil Performa

Program dijalankan di komputer dengan Spec sebagai berikut:
* CPU : Ryzen 9 7950X 16 core 32 thread
* GPU : RTX 5080 16GB VRam

### 1D

| Partisi | Serial (ms) | CUDA (ms) | Hasil CUDA | Error CUDA | Speedup |
|---:|---:|---:|---:|---:|---:|
| 100000 | 0.060553 | 227.247 | 3 | 2.24999e-10 | 0.0003x |
| 1000000 | 0.632835 | 0.123480 | 3 | 2.24754e-12 | 5.12x |
| 1000000000 | 589.874 | 32.3221 | 3 | 5.32907e-14 | 18.25x |
| 10000000000 | 5835.19 | 320.362 | 3 | 4.27658e-13 | 18.21x |
| 100000000000 | 58252.0 | 3240.44 | 3 | 4.13003e-14 | 17.98x |

### 2D

| Partisi | Serial (ms) | CUDA (ms) | Hasil CUDA | Error CUDA | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1000 x 1000 | 0.762251 | 129.587 | 18 | 1.35e-05 | 0.006x |
| 10000 x 10000 | 73.5952 | 4.38116 | 18 | 1.35e-07 | 16.80x |
| 30000 x 30000 | 652.876 | 39.6014 | 18 | 1.50002e-08 | 16.49x |
| 100000 x 100000 | 7275.41 | 449.839 | 18 | 1.34949e-09 | 16.17x |
| 1000000 x 1000000 | - | 45275.4 | 18 | 3.65219e-12 | - |

### 3D

| Partisi | Serial (ms) | CUDA (ms) | Hasil CUDA | Error CUDA | Speedup |
|---:|---:|---:|---:|---:|---:|
| 1000^3 | 762.966 | 190.441 | 80.9999 | 5.65255e-05 | 4.01x |
| 2000^3 | 5992.48 | 459.565 | 81 | 1.0963e-05 | 13.04x |
| 3000^3 | 20169.3 | 1555.44 | 81 | 1.63338e-05 | 12.97x |
| 4000^3 | 47847.4 | 3699.72 | 81 | 4.27541e-07 | 12.93x |

## Analisis Singkat

- CUDA memberi peningkatan performa yang signifikan ketika jumlah partisi besar.
- Untuk ukuran kecil, seperti 1D `100000` dan 2D `1000 x 1000`, versi CUDA lebih lambat karena overhead alokasi memori, transfer data, peluncuran kernel, dan sinkronisasi GPU lebih besar daripada pekerjaan komputasinya.
- Pada 1D ukuran besar, speedup CUDA stabil di sekitar `18x`.
- Pada 2D ukuran besar, speedup CUDA berada di sekitar `16x`.
- Pada 3D, CUDA mencapai speedup sekitar `13x` untuk partisi `2000^3` sampai `4000^3`.
- Error numerik relatif kecil dan hasil integral mendekati nilai analitik pada semua dimensi.

## Catatan

Waktu eksekusi dapat berbeda pada perangkat lain karena dipengaruhi oleh spesifikasi CPU, GPU, versi CUDA, driver, optimasi compiler, dan kondisi runtime.
