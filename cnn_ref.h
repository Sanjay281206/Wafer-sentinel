#ifndef CNN_REF_H
#define CNN_REF_H
#include <stdint.h>
/* map: 4096 die values (0 outside, 1 pass, 2 fail), row-major 64x64. logits: 8 int32 (same numbers as the FPGA). */
void cnn_infer(const uint8_t *map, int32_t *logits);
#endif
