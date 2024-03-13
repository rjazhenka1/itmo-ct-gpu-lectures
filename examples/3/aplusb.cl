kernel void add(global const int *a, global const int *b, global int *c) {
  size_t x = get_global_id(0); 
  c[x] = a[x] + b[x];
}