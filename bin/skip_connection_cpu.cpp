// CPU-only skip connection: adds two int32 binary files element-wise.
// No GPU initialization needed — should complete in <0.1s.
#include <cstdio>
#include <cstdlib>
#include <cstdint>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input1.bin> <input2.bin> <output.bin>\n", argv[0]);
        return 1;
    }

    FILE* f1 = fopen(argv[1], "rb");
    FILE* f2 = fopen(argv[2], "rb");
    if (!f1 || !f2) { fprintf(stderr, "Cannot open input files\n"); return 1; }

    fseek(f1, 0, SEEK_END);
    long size = ftell(f1);
    fseek(f1, 0, SEEK_SET);

    int n = size / sizeof(int32_t);
    int32_t* a = (int32_t*)malloc(size);
    int32_t* b = (int32_t*)malloc(size);

    fread(a, sizeof(int32_t), n, f1);
    fread(b, sizeof(int32_t), n, f2);
    fclose(f1);
    fclose(f2);

    // Element-wise addition (values are small enough that int32 doesn't overflow)
    for (int i = 0; i < n; i++)
        a[i] += b[i];

    FILE* fo = fopen(argv[3], "wb");
    fwrite(a, sizeof(int32_t), n, fo);
    fclose(fo);

    free(a);
    free(b);
    return 0;
}
