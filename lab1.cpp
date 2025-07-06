#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <omp.h>
#include <stdexcept>

void partial_sum_parallel(std::vector<size_t>& arr, size_t num_threads) 
{
    size_t n = arr.size();
    if (n <= 1) return;

    
    size_t chunk_size = 0;
    size_t total_threads = std::min(num_threads, n/2);
    chunk_size = (n + total_threads - 1) / total_threads; 
    total_threads = (n + chunk_size - 1) / chunk_size; 
    



    
#pragma omp parallel num_threads(total_threads)
    {
        size_t thread_id = omp_get_thread_num();
        

        size_t start = thread_id * chunk_size; 
        size_t end = std::min(start + chunk_size, n); 

        for (size_t i = start + 1; i < end; ++i) {
            arr[i] += arr[i - 1];
        }
    

#pragma omp barrier
#pragma omp single
        {
            
            for (size_t block = 1; block < total_threads; ++block) {
                size_t prev_last = block * chunk_size - 1; 
                size_t curr_last = std::min((block + 1) * chunk_size - 1, n - 1);   
                arr[curr_last] += arr[prev_last];
            }
        }



        
        if (thread_id != 0)
        {
            size_t prev_last = thread_id * chunk_size - 1; 
            size_t start = thread_id * chunk_size;        
            size_t end = std::min(start + chunk_size - 1, n - 1); 

            for (size_t i = start; i < end; ++i) {
                arr[i] += arr[prev_last];
            }
        }
    }
}

void partial_sum_parallel(std::vector<size_t>& arr) {
    size_t n = arr.size();
    if (n <= 1) return;

    size_t total_threads = 0;
    size_t chunk_size = 0;



    
#pragma omp parallel
    {
#pragma omp single
        {
            
            total_threads = omp_get_num_threads();
            total_threads = std::min(n/2, total_threads);
            chunk_size = (n + total_threads - 1) / total_threads; 
            total_threads = (n + chunk_size - 1) / chunk_size; 
        }
        size_t thread_id = omp_get_thread_num();
        size_t start = thread_id * chunk_size; 
        size_t end = std::min(start + chunk_size, n); 

        for (size_t i = start + 1; i < end; ++i) {
            arr[i] += arr[i - 1];
        }

    

#pragma omp barrier
#pragma omp single
        {
            
            for (size_t block = 1; block < total_threads; ++block) {
                size_t prev_last = block * chunk_size - 1; 
                size_t curr_last = std::min((block + 1) * chunk_size - 1, n - 1);   
                arr[curr_last] += arr[prev_last];
            }   
        }



    
        if (thread_id != 0)
        {
            size_t prev_last = thread_id * chunk_size - 1; 
            size_t start = thread_id * chunk_size;        
            size_t end = std::min(start + chunk_size - 1, n - 1); 

            for (size_t i = start; i < end; ++i) {
                arr[i] += arr[prev_last];
            }
        }
    }
}

void partial_summ_seq(std::vector<size_t>& arr)
{
    size_t n = arr.size();
    if (n <= 1)
        return;
    for (size_t i = 1; i < n; i++)
    {
        arr[i] += arr[i - 1];
    }
}



size_t find_threshold() {
    size_t M = std::pow(2, 25);
    
    double seq_time, par_time;
    std::vector<size_t> arr;
    std::vector<size_t> arr1;


    while (true) {
        arr.assign(M, 1);
        auto start = std::chrono::high_resolution_clock::now();
        
        partial_summ_seq(arr);
        auto end = std::chrono::high_resolution_clock::now();
        seq_time = std::chrono::duration<double>(end - start).count();

        arr1.assign(M, 1);
        start = std::chrono::high_resolution_clock::now();
        
        partial_sum_parallel(arr1);
        end = std::chrono::high_resolution_clock::now();
        par_time = std::chrono::duration<double>(end - start).count();

        bool flag=false;
        for (int i = 0; i < M; i++)
        {
            
            if (arr[i] != arr1[i])
            {
                flag=true;
                break;
            }
        }
        if (flag)
        {
            for (auto ar : arr1)
                printf("%d", ar);
            break;
        }


        std::cout << "Parallel algorithm time " << par_time << " , sequence " << seq_time << " , M = " << M << std::endl; 

        if (par_time < seq_time) break;
        M *= 2;
    }

    
    std::ofstream file("threshold.txt");
    file << M;
    file.close();

    return M;
}


void partial_sum_hybrid(std::vector<size_t>& arr) {
    size_t n = arr.size();
    if (n <= 1) return;

    
    size_t M;
    std::ifstream file("threshold.txt");
    if (file.is_open()) {
        file >> M;
        file.close();
    }
    else {
        throw std::runtime_error("Не удалось открыть файл threshold.txt");
    }

    if (n >= M) {
        
        partial_sum_parallel(arr);
    }
    else {
        
        partial_summ_seq(arr);
    }
}


void benchmark(size_t N) {
    std::vector<size_t> arr(N, 1);
    std::ofstream file("speedup.csv");
    file << "Threads,Time\n";

    for (size_t threads = 1; threads <= omp_get_max_threads(); ++threads) {
        auto start = std::chrono::high_resolution_clock::now();
        partial_sum_parallel(arr, threads);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(end - start).count();
        file << threads << "," << elapsed << "\n";
        std::cout << threads << "," << elapsed << "\n";
    }

    file.close();
}

int main() {
    try {
        
        std::cout << "Counting threshold M...\n";
        size_t M = find_threshold();
        std::cout << "Threshold meaning M: " << M << "\n";
        
        
        std::cout << "Hybrid algorithm...\n";
        size_t N = 2 * M;
        std::vector<size_t> arr(N, 1);

        partial_sum_hybrid(arr);

        
        std::cout << "Partial summ counting results (first 10):\n";
        for (size_t i = 0; i < std::min(size_t(10), N); ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << "\n";
        
        
        std::cout << "Time counting for speedup diagram...\n";
        benchmark(N);
        std::cout << "Results saved in speedup.csv.\n";
        
    }
    catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }


    return 0;
}
