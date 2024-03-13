#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <assert.h>
#include <stdio.h>

int main() {
  // Получим список ID платформ
  cl_uint platform_count;
  clGetPlatformIDs(0, NULL, &platform_count);
  printf("%u platforms found.\n", platform_count);
  cl_platform_id *platform_ids =
      malloc(platform_count * sizeof(cl_platform_id));
  clGetPlatformIDs(platform_count, platform_ids, NULL);
  cl_platform_id platform_id = platform_ids[0];
  printf("Using first available platform\n");

  // Получим список ID устройств
  cl_uint device_count;
  clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count);
  printf("%u devices found.\n", device_count);
  cl_device_id *device_ids = malloc(device_count * sizeof(cl_device_id));
  clGetDeviceIDs(platform_ids[0], CL_DEVICE_TYPE_ALL, device_count, device_ids,
                 NULL);
  cl_device_id device_id = device_ids[0];
  printf("Using first available device\n");

  // Создадим контекст программы
  cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);

  // Загрузим программу в контекст
  FILE *aplusb_file =
      fopen("../examples/2/aplusb.cl",
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
  printf("Build log:\n%s\n", log);
  free(log);

  // Проверим успешность компиляции
  if (build_err != 0) {
    printf("Could not build program! See log for details.\n");
    return -2;
  }

  printf("Built program successfully!\n");

  // Создадим очередь команд
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, NULL);
  
  // Получим идентификатор кернела
  cl_kernel kernel = clCreateKernel(program, "add", NULL);

  // Выделим память на девайсе и поставим ее как аргументы кернела
  cl_mem a_mem =
      clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
  cl_mem b_mem =
      clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int), NULL, NULL);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
  cl_mem c_mem =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int), NULL, NULL);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);

  // Поставим в очередь запись аргументов
  cl_int a = 2, b = 3;
  clEnqueueWriteBuffer(command_queue, a_mem, CL_FALSE, 0, sizeof(cl_int), &a, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, b_mem, CL_FALSE, 0, sizeof(cl_int), &b, 0, NULL,
                       NULL);

  // Поставим в очередь запуск кернела
  size_t zero = 0, one = 1;
  clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &one, NULL, 0, NULL, NULL);
  
  // Поставим в очередь чтение ответа (в блокирующем режиме, что запустит нашу программу)
  cl_int c;
  clEnqueueReadBuffer(command_queue, c_mem, CL_TRUE, 0, sizeof(cl_int), &c, 0, NULL,
                      NULL);

  // Проверим
  printf("Executed %d + %d successfully, got %d (expected %d).\n", a, b, c, a + b);
  assert(a + b == c);

  // Приберемся напоследок
  clReleaseMemObject(a_mem);
  clReleaseMemObject(b_mem);
  clReleaseMemObject(c_mem);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  free(platform_ids);
  free(device_ids);

  return 0;
}