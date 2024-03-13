#include <CL/cl_platform.h>
#include <bits/time.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <assert.h>
#include <omp.h>
#include <stdio.h>

#define N 4000
#define K 3200
#define M 2800
#define EPS 1e-6

struct cl_result {
  cl_float *c;
  uint64_t time_us;
};

struct mp_result {
  cl_float *c;
  uint64_t time_us;
};

int find_diff(cl_float *l, cl_float *r) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int index = i * N + j;
      if (fabs(l[index] - r[index]) > EPS) {
        return index;
      }
    }
  }

  return -1;
}

struct mp_result run_openmp(const cl_float *a, const cl_float *b) {
  cl_float *c = malloc(sizeof(cl_float) * M * N);
  for (int i = 0; i < M * N; i++) {
    c[i] = 0;
  }

  int n, m, k;
  omp_set_num_threads(omp_get_num_procs());
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);

#pragma omp parallel for private(n, m, k) shared(a, b, c)
  for (n = 0; n < N; ++n) {
    for (m = 0; m < M; ++m) {
      cl_float sum = 0;
      for (k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
  struct mp_result result;
  result.c = c;
  result.time_us = (end.tv_sec - start.tv_sec) * 1000000 +
                   (end.tv_nsec - start.tv_nsec) / 1000;
  return result;
}

struct cl_result run_device(cl_device_id device_id, const cl_float *a,
                            const cl_float *b);


int run_platform(cl_platform_id platform_id) {
  // Получим название платформы
  size_t name_size;
  clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &name_size);
  char *name = malloc(name_size + 1);
  clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, name_size, name, NULL);
  name[name_size] = 0;
  printf("==========================\n");
  printf("Running on platform: %s\n", name);
  free(name);

  // Получим список ID устройств
  cl_uint device_count;
  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
  printf("%u devices found.\n", device_count);
  cl_device_id *device_ids = malloc(device_count * sizeof(cl_device_id));
  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, device_count, device_ids,
                 NULL);

  for (int i = 0; i < device_count; i++) {
    cl_float *a = malloc(sizeof(cl_float) * M * K);
    cl_float *b = malloc(sizeof(cl_float) * K * N);
    for (int i = 0; i < M * K; i++) {
      a[i] = (rand() % 10) / 2 - 5;
    }

    for (int i = 0; i < K * N; i++) {
      b[i] = (rand() % 10) / 2 - 5;
    }

    struct cl_result cl_result = run_device(device_ids[i], a, b);
    struct mp_result mp_result = run_openmp(a, b);

    // Проверим
    /*
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        printf("%.1f vs %.1f, ", cl_result.c[i * N + j], mp_result.c[i * N +
    j]);
      }
      printf("\n");
    }
    */

    assert(find_diff(cl_result.c, mp_result.c) == -1);
    printf("multiplied matrices successfully!\n");

    // Выведем скорость и посчитаем FLOPS:

    cl_ulong operation_count = 2UL * M * N * K;
    printf("OpenMP took %lu ms (%.1f GFLOPS).\n", mp_result.time_us, (double) operation_count / mp_result.time_us / 1000);
    printf("OpenCL took %lu ms (%.1f GFLOPS).\n", cl_result.time_us, (double) operation_count / cl_result.time_us / 1000);


    free(cl_result.c);
    free(mp_result.c);
  }

  free(device_ids);

  return 0;
}

struct cl_result run_device(cl_device_id device_id, const cl_float *a,
                            const cl_float *b) {
  struct cl_result result;
  result.c = NULL;

  // Получим название устройства
  size_t name_size;
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &name_size);
  char *name = malloc(name_size + 1);
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, name_size, name, NULL);
  name[name_size] = 0;
  printf("--------------------------\n");
  printf("Running on device: %s\n", name);
  free(name);

  // Создадим контекст программы
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

  // Загрузим программу в контекст
  FILE *aplusb_file =
      fopen("../examples/4/matrix-multiply.cl",
            "rb");                 // Откроем файл
  fseek(aplusb_file, 0, SEEK_END); // Поставим указатель на конец
  size_t aplusb_size = ftell(aplusb_file); // Запишем позицию конца
  fseek(aplusb_file, 0, SEEK_SET); // Вернемся в начало файла

  char *aplusb_source =
      malloc(aplusb_size + 1); // Выделим буфер под содержимое файла
  fread(aplusb_source, aplusb_size, 1, aplusb_file); // Считаем содержимое файла
  fclose(aplusb_file);
  aplusb_source[aplusb_size] = 0; // Строки в Си оканчиваются на ноль!

  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&aplusb_source,
                                (const size_t *)&aplusb_size, NULL);

  // Скомпилируем программу
  cl_int build_err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);

  // Получим логи
  size_t log_size;
  clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL,
                        &log_size);
  char *log = malloc(log_size);
  clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log,
                        NULL);
  printf("build log:\n%s\n", log);
  free(log);

  // Проверим успешность компиляции
  if (build_err != 0) {
    printf("Could not build program! See log for details.\n");
    return result;
  }

  printf("built program successfully!\n");

  // Создадим очередь команд
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

  // Получим идентификатор кернела
  cl_kernel kernel = clCreateKernel(program, "mmul", NULL);

  // Выделим память на девайсе и поставим ее как аргументы кернела
  cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(cl_float) * M * K, NULL, NULL);
  cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(cl_float) * K * N, NULL, NULL);
  cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                sizeof(cl_float) * M * N, NULL, NULL);

  cl_uint n = N;
  cl_uint k = K;
  cl_uint m = M;
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
  clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
  clSetKernelArg(kernel, 4, sizeof(cl_uint), &k);
  clSetKernelArg(kernel, 5, sizeof(cl_uint), &m);

  clEnqueueWriteBuffer(command_queue, a_mem, CL_FALSE, 0,
                       sizeof(cl_float) * M * K, a, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, b_mem, CL_FALSE, 0,
                       sizeof(cl_float) * K * N, b, 0, NULL, NULL);

  // Поставим в очередь запуск кернела
  size_t global_work_size[] = {N, M};
  cl_event event;
  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL,
                         0, NULL, &event);

  // Поставим в очередь чтение ответа (в блокирующем режиме, что запустит нашу
  // программу)
  cl_float *c = malloc(sizeof(cl_float) * M * N);
  clEnqueueReadBuffer(command_queue, c_mem, CL_TRUE, 0,
                      sizeof(cl_float) * M * N, c, 0, NULL, NULL);

  // Посчитаем время исполнения
  clWaitForEvents(1, &event);
  clFinish(command_queue);

  cl_ulong time_start, time_end;
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start),
                          &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end),
                          &time_end, NULL);

  result.time_us = (time_end - time_start) / 1000;

  // Приберемся напоследок
  clReleaseMemObject(a_mem);
  clReleaseMemObject(b_mem);
  clReleaseMemObject(c_mem);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  result.c = c;
  return result;
}

int main() {
  // Получим список ID платформ
  cl_uint platform_count;
  clGetPlatformIDs(0, NULL, &platform_count);
  printf("%u platforms found.\n", platform_count);
  cl_platform_id *platform_ids =
      malloc(platform_count * sizeof(cl_platform_id));
  clGetPlatformIDs(platform_count, platform_ids, NULL);

  for (int i = 0; i < platform_count; i++) {
    run_platform(platform_ids[i]);
  }

  free(platform_ids);
  return 0;
}