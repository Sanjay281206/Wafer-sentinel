// AXI4-Lite wrapper for wafer_cnn_accel (use as a module reference in the Vivado block design).
// Register map (byte offsets, 8 KB window):
//   0x0000 CTRL    write bit0 = 1 -> start inference
//   0x0004 STATUS  bit0 busy, bit1 done (stays 1 until the next start)
//   0x0008 CYCLES  clock cycles of the last inference
//   0x0010 + 4*i   LOGIT i (signed int32), i = 0..7
//   0x1000 + 4*j   map word j (j = 0..1023): four die values (0 = outside, 1 = pass, 2 = fail), pixel 4j in the low byte
`timescale 1ns/1ps
module wafer_cnn_axil #(
    parameter WFILE = "weights.hex", BFILE = "bias.hex", MFILE = "mult.hex", SFILE = "shift.hex"
)(
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire        s_axi_aclk,
    input  wire        s_axi_aresetn,
    input  wire [12:0] s_axi_awaddr,
    input  wire        s_axi_awvalid,
    output reg         s_axi_awready,
    input  wire [31:0] s_axi_wdata,
    input  wire [3:0]  s_axi_wstrb,
    input  wire        s_axi_wvalid,
    output reg         s_axi_wready,
    output wire [1:0]  s_axi_bresp,
    output reg         s_axi_bvalid,
    input  wire        s_axi_bready,
    input  wire [12:0] s_axi_araddr,
    input  wire        s_axi_arvalid,
    output reg         s_axi_arready,
    output reg  [31:0] s_axi_rdata,
    output wire [1:0]  s_axi_rresp,
    output reg         s_axi_rvalid,
    input  wire        s_axi_rready
);
    assign s_axi_bresp = 2'b00;
    assign s_axi_rresp = 2'b00;

    reg [12:0] waddr, raddr; reg [31:0] wdat; reg do_write;
    reg start_p, img_we; reg [9:0] img_waddr; reg [31:0] img_wdata;
    wire busy, done; wire [31:0] res_logit, cycles;
    wire [2:0] res_idx = (raddr - 13'h0010) >> 2;

    wafer_cnn_accel #(.WFILE(WFILE), .BFILE(BFILE), .MFILE(MFILE), .SFILE(SFILE)) core (
        .clk(s_axi_aclk), .rst_n(s_axi_aresetn), .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
        .start(start_p), .busy(busy), .done(done), .res_idx(res_idx), .res_logit(res_logit), .cycle_count(cycles));

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 0; s_axi_wready <= 0; s_axi_bvalid <= 0; do_write <= 0; start_p <= 0; img_we <= 0;
            s_axi_arready <= 0; s_axi_rvalid <= 0; s_axi_rdata <= 0; waddr <= 0; raddr <= 0; wdat <= 0;
            img_waddr <= 0; img_wdata <= 0;
        end else begin
            // ---- write channel: accept address and data together
            if (!s_axi_awready && s_axi_awvalid && s_axi_wvalid && !s_axi_bvalid && !do_write) begin
                s_axi_awready <= 1; s_axi_wready <= 1; waddr <= s_axi_awaddr; wdat <= s_axi_wdata; do_write <= 1;
            end else begin
                s_axi_awready <= 0; s_axi_wready <= 0; do_write <= 0;
            end
            if (do_write) s_axi_bvalid <= 1; else if (s_axi_bvalid && s_axi_bready) s_axi_bvalid <= 0;
            start_p <= do_write && (waddr[12:2] == 11'd0) && wdat[0];
            img_we  <= do_write && waddr[12];
            img_waddr <= waddr[11:2];
            img_wdata <= wdat;
            // ---- read channel
            if (!s_axi_arready && s_axi_arvalid && !s_axi_rvalid) begin s_axi_arready <= 1; raddr <= s_axi_araddr; end
            else s_axi_arready <= 0;
            if (s_axi_arready) begin
                s_axi_rvalid <= 1;
                if (raddr == 13'h0004)                          s_axi_rdata <= {30'd0, done, busy};
                else if (raddr == 13'h0008)                     s_axi_rdata <= cycles;
                else if (raddr >= 13'h0010 && raddr < 13'h0030) s_axi_rdata <= res_logit;
                else                                            s_axi_rdata <= 32'd0;
            end else if (s_axi_rvalid && s_axi_rready) s_axi_rvalid <= 0;
        end
    end
endmodule
