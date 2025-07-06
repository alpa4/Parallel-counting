#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>

#define TERMINATE_MSG -1
#define FREE_PROCESS_MSG -2

// Простейшие функции конвейера
float f1(float x) { return x + 1.0; }
float f2(float x) { return x * 2.0; }
float f3(float x) { return x - 3.0; }

// Функции для удобного изменения порядка вычислений
typedef float (*func_ptr)(float);
std::vector<func_ptr> functions = { f1, f2, f3 };

using namespace std;

void process_zero(int data_size, float data[], int broker_rank) {
    for (int i = 0; i < data_size; i++) {
        float msg[2] = { static_cast<float>(i), data[i] };
        MPI_Send(msg, 2, MPI_FLOAT, broker_rank, 0, MPI_COMM_WORLD);
        cout << "0 процесс отправил переменную " << msg[0] << " со значением " << msg[1] << endl;
    }
}

void process_worker(int stage, int broker_in, int broker_out) {
    while (1) {
        float msg[2];
        MPI_Recv(&msg, 2, MPI_FLOAT, broker_in, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (static_cast<int>(msg[0]) == TERMINATE_MSG) break;

        cout << "переменная " << msg[0] << " со значением " << msg[1] <<  "стадия" << endl;

        msg[1] = functions[stage](msg[1]);
        MPI_Send(msg, 2, MPI_FLOAT, broker_out, 0, MPI_COMM_WORLD);

        float free_msg[2] = { static_cast<float>(FREE_PROCESS_MSG), static_cast<float>(broker_in) };
        MPI_Send(free_msg, 2, MPI_FLOAT, broker_in, 0, MPI_COMM_WORLD);
    }
}

void process_broker(std::vector<int> workers, int data_size) {
    std::vector<std::pair<int, float>> queue;
    std::vector<int> free_workers = workers;
    int received_count = 0;

    MPI_Status st;

    // Получаем все данные
    while (received_count < data_size || !queue.empty() || free_workers.size() < workers.size()) {
        float msg[2];
        MPI_Recv(&msg, 2, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);

        if (static_cast<int>(msg[0]) == FREE_PROCESS_MSG) {
            free_workers.push_back(st.MPI_SOURCE);
        }
        else {
            queue.push_back({ static_cast<int>(msg[0]), msg[1] });
            received_count++;
        }

        // Распределяем задачи среди свободных рабочих
        while (!queue.empty() && !free_workers.empty()) {
            int worker = free_workers.back();
            free_workers.pop_back();

            auto task = queue.front();
            queue.erase(queue.begin());
            float send_msg[2] = { static_cast<float>(task.first), task.second };
            MPI_Send(send_msg, 2, MPI_FLOAT, worker, 0, MPI_COMM_WORLD);
        }
    }

    // Дожидаемся освобождения всех рабочих
    while (free_workers.size() < workers.size()) {
        float msg[2];
        MPI_Recv(&msg, 2, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);

        if (static_cast<int>(msg[0]) == FREE_PROCESS_MSG) {
            free_workers.push_back(st.MPI_SOURCE);
        }
    }

    // Отправляем всем рабочим процессам сообщение о завершении работы
    for (int w : workers) {
        float term_msg[2] = { static_cast<float>(TERMINATE_MSG), static_cast<float>(TERMINATE_MSG) };
        MPI_Send(term_msg, 2, MPI_FLOAT, w, 0, MPI_COMM_WORLD);
    }
}

void process_sum(int total_count) {
    std::vector<float> results(total_count);
    float sum = 0.0;
    for (int i = 0; i < total_count; i++) {
        float msg[2];
        MPI_Recv(&msg, 2, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        results[static_cast<int>(msg[0])] = msg[1];
        sum += msg[1];
    }
    for (int i = 0; i < total_count; i++) {
        printf("Result[%d] = %.2f\n", i, results[i]);
    }
    printf("Total Sum: %.2f\n", sum);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int data_size = 5;
    float data[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
    int distribution[] = { 1, 2, 3 };
    int num_stages = functions.size();

    if (rank == 0) {
        process_zero(data_size, data, 1);
    }
    else if (rank == size - 1) {
        process_sum(data_size);
    }
    else {
        int broker_rank = 1;
        int worker_start = 2;

        for (int stage = 0; stage < num_stages; stage++) {
            int num_workers = distribution[stage];
            std::vector<int> workers;
            for (int i = 0; i < num_workers; i++) {
                workers.push_back(worker_start + i);
            }

            if (rank == broker_rank) {
                process_broker(workers, data_size);
            }
            else if (std::find(workers.begin(), workers.end(), rank) != workers.end()) {
                process_worker(stage, broker_rank, workers.back()+1);
            }

            broker_rank += num_workers + 1;
            worker_start += num_workers;
        }
    }

    MPI_Finalize();
    return 0;
}