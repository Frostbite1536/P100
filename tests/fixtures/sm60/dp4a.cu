extern "C" __global__ void forbidden_dot(int a, int b, int *out) {
    *out = __dp4a(a, b, 0);
}
