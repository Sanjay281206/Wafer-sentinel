/* ZCU104 bare-metal bring-up firmware (Vitis, standalone BSP, Cortex-A53 core 0, PS UART0): serial protocol between a host
   computer (USB-UART) and the CNN accelerator. Same protocol as board/zcu_server.py, so bench/bench_fpga.py works unchanged:
   python bench_fpga.py --target /dev/ttyUSB1 ...   (UART_BAUD below, default 921600: about 0.05 s per wafer on the wire)
   Request : 0xA5, cmd, 4096 map bytes (values 0..2) [, uint32 n for cmd 3]      (little-endian)
     cmd 1 = run on the FPGA accelerator, cmd 2 = run the C reference on the Cortex-A53,
     cmd 3 = run the FPGA accelerator n times back to back (sustained speed / power measurement)
   Response: 0xA5, status (0 ok, 1 timeout), 8 x int32 logits, uint32 accelerator cycles, uint64 A53 timer ticks,
             uint32 timer ticks per second                                        (50 bytes) */
#include <stdint.h>
#include "xil_io.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xtime_l.h"
#include "xuartps.h"
#include "xuartps_hw.h"
#include "cnn_ref.h"

#ifndef ACCEL_BASE
#define ACCEL_BASE 0xA0000000u        /* ZynqMP: first slave behind M_AXI_HPM0_FPD (was 0x43C00000 on the Zynq-7000 ZedBoard).
                                         Check Vivado's Address Editor / the build log line "accelerator ... offset". */
#endif
#ifndef UART_BAUD
#define UART_BAUD 921600u             /* host must use the same value (bench_fpga.py --baud, app sidebar) */
#endif
#ifndef UART_DEVICE_ID
#define UART_DEVICE_ID 0              /* PS UART0 */
#endif
#define REG_CTRL   0x0000
#define REG_STATUS 0x0004
#define REG_CYCLES 0x0008
#define REG_LOGIT  0x0010
#define REG_IMG    0x1000

static uint8_t map[4096];

static uint8_t rx(void) { return XUartPs_RecvByte(STDIN_BASEADDRESS); }
static void tx(uint8_t b) { XUartPs_SendByte(STDOUT_BASEADDRESS, b); }
static void tx32(uint32_t v) { for (int i = 0; i < 4; i++) tx((uint8_t)(v >> (8 * i))); }
static void tx64(uint64_t v) { for (int i = 0; i < 8; i++) tx((uint8_t)(v >> (8 * i))); }

/* Raise the PS UART baud rate above the 115200 default (4 KB per wafer: 0.36 s at 115200, 0.045 s at 921600). */
static void uart_setup(void) {
    static XUartPs uart;
    XUartPs_Config *cfg = XUartPs_LookupConfig(UART_DEVICE_ID);
    if (cfg == NULL || (UINTPTR)cfg->BaseAddress != (UINTPTR)STDIN_BASEADDRESS) return;   /* keep the BSP default */
    if (XUartPs_CfgInitialize(&uart, cfg, cfg->BaseAddress) != XST_SUCCESS) return;
    XUartPs_SetBaudRate(&uart, UART_BAUD);
    uint32_t cr = XUartPs_ReadReg(cfg->BaseAddress, XUARTPS_CR_OFFSET);
    cr &= ~(uint32_t)(XUARTPS_CR_TX_DIS | XUARTPS_CR_RX_DIS);
    cr |= XUARTPS_CR_TX_EN | XUARTPS_CR_RX_EN;
    XUartPs_WriteReg(cfg->BaseAddress, XUARTPS_CR_OFFSET, cr);
}

static void load_image(void) {
    for (int i = 0; i < 1024; i++) {
        uint32_t w = (uint32_t)map[4 * i] | ((uint32_t)map[4 * i + 1] << 8) |
                     ((uint32_t)map[4 * i + 2] << 16) | ((uint32_t)map[4 * i + 3] << 24);
        Xil_Out32(ACCEL_BASE + REG_IMG + 4 * i, w);
    }
}

/* start the accelerator and wait for it (waits for busy to rise, then fall, so a stale "done" is never used) */
static int run_hw(void) {
    uint32_t guard = 0;
    Xil_Out32(ACCEL_BASE + REG_CTRL, 1);
    while (!(Xil_In32(ACCEL_BASE + REG_STATUS) & 1)) if (++guard > 1000000u) return 1;
    guard = 0;
    while (Xil_In32(ACCEL_BASE + REG_STATUS) & 1) if (++guard > 400000000u) return 1;
    return 0;
}

int main(void) {
    Xil_DCacheEnable();                /* normally already on; keeps the C baseline realistic */
    uart_setup();
    for (;;) {
        int32_t logits[8] = {0};
        uint32_t cycles = 0, n = 1, status = 0;
        XTime t0, t1;
        while (rx() != 0xA5) { }
        uint8_t cmd = rx();
        for (int i = 0; i < 4096; i++) map[i] = rx();
        if (cmd == 3) { n = 0; for (int i = 0; i < 4; i++) n |= (uint32_t)rx() << (8 * i); }
        if (cmd == 1 || cmd == 3) {
            load_image();
            XTime_GetTime(&t0);
            for (uint32_t k = 0; k < n && !status; k++) status = run_hw();
            XTime_GetTime(&t1);
            cycles = Xil_In32(ACCEL_BASE + REG_CYCLES);
            for (int i = 0; i < 8; i++) logits[i] = (int32_t)Xil_In32(ACCEL_BASE + REG_LOGIT + 4 * i);
        } else {
            XTime_GetTime(&t0);
            cnn_infer(map, logits);
            XTime_GetTime(&t1);
        }
        tx(0xA5); tx((uint8_t)status);
        for (int i = 0; i < 8; i++) tx32((uint32_t)logits[i]);
        tx32(cycles); tx64((uint64_t)(t1 - t0)); tx32((uint32_t)COUNTS_PER_SECOND);
    }
    return 0;
}
