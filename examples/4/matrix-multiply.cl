kernel void mmul(global const float *a, global const float *b, global float *c, const uint n, const uint k, const uint m) {
  uint x = get_global_id(0); 
  uint y = get_global_id(1);
  float sum = 0;
  for (uint i = 0; i < k; i++) {
    sum += a[y * k + i] * b[i * n + x];
  }
  c[y * n + x] = sum;
}