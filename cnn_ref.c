/* Plain-C integer reference of the WaferCNN: same arithmetic as the FPGA accelerator and intmodel.py (bit-exact).
   Used on the ZCU104's Cortex-A53 as the software baseline (bare-metal here, libcnnref.so under Linux). Not optimised on purpose. */
#include "cnn_ref.h"
#include "weights.h"

static uint8_t buf_a[16 * 64 * 64];     /* current activations (channel-major) */
static uint8_t buf_b[16 * 64 * 64];     /* conv output before pooling */

static const int IC[4] = {3, 16, 32, 64}, OC[4] = {16, 32, 64, 64};
static const int8_t *const WT[4] = {W0, W1, W2, W3};
static const int32_t *const BI[4] = {B0, B1, B2, B3};
static const uint16_t *const MU[4] = {M0, M1, M2, M3};
static const uint8_t *const SH[4] = {S0, S1, S2, S3};

void cnn_infer(const uint8_t *map, int32_t *logits) {
    int i, l, o, c, y, x, ky, kx, h = 64;
    for (i = 0; i < 4096; i++)
        for (c = 0; c < 3; c++) buf_a[c * 4096 + i] = (map[i] == c);
    for (l = 0; l < 4; l++) {
        const int ic = IC[l], oc = OC[l];
        for (o = 0; o < oc; o++)
            for (y = 0; y < h; y++)
                for (x = 0; x < h; x++) {
                    int32_t acc = 0;
                    for (c = 0; c < ic; c++)
                        for (ky = 0; ky < 3; ky++) {
                            int iy = y + ky - 1;
                            if (iy < 0 || iy >= h) continue;
                            for (kx = 0; kx < 3; kx++) {
                                int ix = x + kx - 1;
                                if (ix < 0 || ix >= h) continue;
                                acc += (int32_t)buf_a[(c * h + iy) * h + ix] * WT[l][((o * ic + c) * 3 + ky) * 3 + kx];
                            }
                        }
                    int64_t t = (int64_t)acc + BI[l][o];
                    int64_t v = (t * MU[l][o] + ((int64_t)1 << (SH[l][o] - 1))) >> SH[l][o];
                    buf_b[(o * h + y) * h + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
        if (l < 3) {                                    /* 2x2 max pool */
            int hp = h / 2;
            for (o = 0; o < oc; o++)
                for (y = 0; y < hp; y++)
                    for (x = 0; x < hp; x++) {
                        const uint8_t *p = &buf_b[(o * h + 2 * y) * h + 2 * x];
                        uint8_t m = p[0];
                        if (p[1] > m) m = p[1];
                        if (p[h] > m) m = p[h];
                        if (p[h + 1] > m) m = p[h + 1];
                        buf_a[(o * hp + y) * hp + x] = m;
                    }
            h = hp;
        }
    }
    {   /* global average pool over the 8x8 map, then the linear layer */
        int32_t gap[64];
        for (o = 0; o < 64; o++) {
            int32_t s = 0;
            for (i = 0; i < 64; i++) s += buf_b[o * 64 + i];
            gap[o] = (s + 32) >> 6;
        }
        for (o = 0; o < 8; o++) {
            int32_t acc = BFC[o];
            for (c = 0; c < 64; c++) acc += gap[c] * WFC[o * 64 + c];
            logits[o] = acc;
        }
    }
}
