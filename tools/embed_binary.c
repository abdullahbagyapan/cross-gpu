#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int write_header(FILE *output)
{
    return fputs("/* Generated from a binary kernel module; do not edit. */\n"
                 "#include <cstddef>\n\n"
                 "extern \"C\" {\n"
                 "extern const unsigned char crossgpu_levelzero_kernels_spv[] = {\n",
                 output) == EOF ? -1 : 0;
}

int main(int argc, char **argv)
{
    FILE *input;
    FILE *output;
    unsigned char buffer[4096];
    size_t count;
    size_t offset = 0;

    if (argc != 3) {
        fprintf(stderr, "usage: %s INPUT OUTPUT\n", argv[0]);
        return 2;
    }
    input = fopen(argv[1], "rb");
    if (input == NULL) {
        fprintf(stderr, "%s: cannot open %s: %s\n", argv[0], argv[1], strerror(errno));
        return 1;
    }
    output = fopen(argv[2], "wb");
    if (output == NULL) {
        fprintf(stderr, "%s: cannot open %s: %s\n", argv[0], argv[2], strerror(errno));
        fclose(input);
        return 1;
    }
    if (write_header(output) != 0)
        goto write_error;

    while ((count = fread(buffer, 1, sizeof(buffer), input)) != 0) {
        size_t i;
        for (i = 0; i < count; i++, offset++) {
            if (offset % 12 == 0 && fputs("    ", output) == EOF)
                goto write_error;
            if (fprintf(output, "0x%02x,", (unsigned int)buffer[i]) < 0)
                goto write_error;
            if (offset % 12 == 11 && fputc('\n', output) == EOF)
                goto write_error;
            else if (offset % 12 != 11 && fputc(' ', output) == EOF)
                goto write_error;
        }
    }
    if (ferror(input)) {
        fprintf(stderr, "%s: cannot read %s\n", argv[0], argv[1]);
        fclose(input);
        fclose(output);
        return 1;
    }
    if (offset == 0 ||
        fputs("\n};\nextern const size_t crossgpu_levelzero_kernels_spv_size = "
              "sizeof(crossgpu_levelzero_kernels_spv);\n}\n", output) == EOF)
        goto write_error;
    if (fclose(input) != 0 || fclose(output) != 0) {
        fprintf(stderr, "%s: failed to close output\n", argv[0]);
        return 1;
    }
    return 0;

write_error:
    fprintf(stderr, "%s: cannot write %s\n", argv[0], argv[2]);
    fclose(input);
    fclose(output);
    return 1;
}
