// WaferSentinel CNN accelerator (INT8, bit-exact with intmodel.py). Verilog-2001, one clock.
// 3 x 64 x 64 one-hot map -> 4 conv layers (16,32,64,64 ch) -> global average -> linear -> 8 int32 logits.
// P = 32 output-channel lanes; one multiply-accumulate per lane per clock.
`timescale 1ns/1ps
module wafer_cnn_accel #(
    parameter WFILE = "weights.hex", BFILE = "bias.hex", MFILE = "mult.hex", SFILE = "shift.hex"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        img_we,          // write 4 map pixels (one per byte, values 0..2) into word img_waddr
    input  wire [9:0]  img_waddr,
    input  wire [31:0] img_wdata,
    input  wire        start,           // one-clock pulse
    output reg         busy,
    output reg         done,            // stays high until the next start
    input  wire [2:0]  res_idx,
    output wire [31:0] res_logit,       // signed int32 logit res_idx
    output reg  [31:0] cycle_count      // clock cycles of the last inference
);
    localparam P  = 32;
    localparam NW = 1963;               // weight words: 27+144+288+288+576+576+64
    localparam NG = 7;                  // lane groups: L1, L2, L3a, L3b, L4a, L4b, FC

    // ------------------------------------------------------------ memories
    reg [P*8-1:0] wrom [0:NW-1];
    reg [31:0]    bias_rom  [0:NG*P-1];
    reg [15:0]    mult_rom  [0:NG*P-1];
    reg [7:0]     shift_rom [0:NG*P-1];
    initial begin
        $readmemh(WFILE, wrom); $readmemh(BFILE, bias_rom); $readmemh(MFILE, mult_rom); $readmemh(SFILE, shift_rom);
    end
    reg [31:0]    img  [0:1023];
    reg [P*8-1:0] bufA [0:1023];        // layer 1 output, later layer 3 output
    reg [P*8-1:0] bufB [0:255];         // layer 2 output
    always @(posedge clk) if (img_we) img[img_waddr] <= img_wdata;

    // ------------------------------------------------------------ control registers
    localparam S_IDLE = 4'd0, S_INIT = 4'd1, S_PREP = 4'd2, S_ISSUE = 4'd3, S_R1 = 4'd4, S_R2 = 4'd5,
               S_R3 = 4'd6, S_UPD = 4'd7, S_GAP = 4'd8, S_FCST = 4'd9;
    reg [3:0] state;
    reg [2:0] layer;                    // 0..3 conv, 4 = linear
    reg       g;                        // lane group inside the layer
    reg [5:0] py, px;
    reg [1:0] sub;                      // position inside the 2x2 pooling window
    reg [9:0] kc;                       // multiply-accumulate counter
    reg [6:0] ic;
    reg [1:0] ky, kx;
    reg [2*P*8-1:0] fcin_flat;           // 64 pooled channels x 8 bit
    reg [255:0]     logit_flat;          // 8 logits x 32 bit

    // layer table
    reg [6:0] ic_n; reg [2:0] logw; reg pooled; reg [1:0] ng; reg [2:0] gg0; reg [9:0] Kmax;
    always @* begin
        case (layer)
            3'd0:    begin ic_n = 3;  logw = 6; pooled = 1; ng = 1; gg0 = 0; Kmax = 10'd27;  end
            3'd1:    begin ic_n = 16; logw = 5; pooled = 1; ng = 1; gg0 = 1; Kmax = 10'd144; end
            3'd2:    begin ic_n = 32; logw = 4; pooled = 1; ng = 2; gg0 = 2; Kmax = 10'd288; end
            3'd3:    begin ic_n = 64; logw = 3; pooled = 0; ng = 2; gg0 = 4; Kmax = 10'd576; end
            default: begin ic_n = 0;  logw = 0; pooled = 0; ng = 1; gg0 = 6; Kmax = 10'd64;  end
        endcase
    end
    wire [2:0] gg = gg0 + {2'b00, g};
    reg [10:0] wb;
    always @* begin
        case (gg)
            3'd0: wb = 11'd0;    3'd1: wb = 11'd27;   3'd2: wb = 11'd171;  3'd3: wb = 11'd459;
            3'd4: wb = 11'd747;  3'd5: wb = 11'd1323; default: wb = 11'd1899;
        endcase
    end
    wire [6:0] wlim = pooled ? (7'd1 << (logw - 3'd1)) : (7'd1 << logw);

    // ------------------------------------------------------------ address generation (stage 0)
    wire [6:0] ycv = pooled ? ({py, 1'b0} | {6'b0, sub[1]}) : {1'b0, py};
    wire [6:0] xcv = pooled ? ({px, 1'b0} | {6'b0, sub[0]}) : {1'b0, px};
    wire signed [8:0] iy = $signed({2'b00, ycv}) + $signed({7'b0, ky}) - 9'sd1;
    wire signed [8:0] ix = $signed({2'b00, xcv}) + $signed({7'b0, kx}) - 9'sd1;
    wire signed [8:0] hw = 9'sd1 << logw;
    wire pad_c = iy[8] | ix[8] | (iy >= hw) | (ix >= hw);
    wire [11:0] pix    = ({6'b0, iy[5:0]} << logw) + {6'b0, ix[5:0]};
    wire [11:0] a_addr = ({5'b0, ic[6:5]} << {logw, 1'b0}) + pix;
    wire [10:0] w_addr = (kc < Kmax) ? (wb + {1'b0, kc}) : wb;

    reg [31:0] img_q; reg [1:0] sel_q;
    reg [P*8-1:0] a_q_A, a_q_B, w_q;
    reg [7:0] fc_a_q;
    reg iss_q, pad_q; reg [6:0] ic_q; reg [2:0] layer_q;
    always @(posedge clk) begin
        img_q   <= img[pix[11:2]];
        sel_q   <= pix[1:0];
        a_q_A   <= bufA[a_addr[9:0]];
        a_q_B   <= bufB[a_addr[7:0]];
        w_q     <= wrom[w_addr];
        fc_a_q  <= fcin_flat[{kc[5:0], 3'b000} +: 8];
        iss_q   <= (state == S_ISSUE) && (kc < Kmax);
        pad_q   <= pad_c;
        ic_q    <= ic;
        layer_q <= layer;
    end

    // ------------------------------------------------------------ stage 1: activation select
    wire [7:0] actA = a_q_A[{ic_q[4:0], 3'b000} +: 8];
    wire [7:0] actB = a_q_B[{ic_q[4:0], 3'b000} +: 8];
    reg  [7:0] act;
    always @* begin
        case (layer_q)
            3'd0:         act = (!pad_q && (img_q[{sel_q, 3'b000} +: 2] == ic_q[1:0])) ? 8'd1 : 8'd0;
            3'd1, 3'd3:   act = pad_q ? 8'd0 : actA;
            3'd2:         act = pad_q ? 8'd0 : actB;
            default:      act = fc_a_q;
        endcase
    end
    wire signed [8:0] act_s = $signed({1'b0, act});
    wire mac_v = iss_q;

    // ------------------------------------------------------------ P lanes
    wire [P*8-1:0]  pmax_flat, gap0_flat, gap1_flat;
    wire [P*32-1:0] t_flat;
    genvar l;
    generate for (l = 0; l < P; l = l + 1) begin : lane
        wire signed [7:0]  wl = w_q[l*8 +: 8];
        wire signed [31:0] bl = bias_rom[gg*P + l];
        wire [15:0]        ml = mult_rom[gg*P + l];
        wire [7:0]         sl = shift_rom[gg*P + l];
        reg signed [31:0] acc;
        reg signed [31:0] t;
        reg signed [47:0] prod;
        reg [7:0]  q;
        reg [7:0]  pmax;
        reg [15:0] gsum0, gsum1;
        wire signed [16:0] mulr = act_s * wl;
        always @(posedge clk) begin
            if (state == S_PREP) acc <= 32'sd0;
            else if (mac_v)      acc <= acc + mulr;
        end
        always @(posedge clk) if (state == S_R1) t <= acc + bl;
        always @(posedge clk) if (state == S_R2) prod <= t * $signed({1'b0, ml});
        wire signed [47:0] rnd = prod + (48'sd1 <<< (sl - 8'd1));
        wire signed [47:0] shv = rnd >>> sl;
        always @(posedge clk) if (state == S_R3) q <= (shv < 0) ? 8'd0 : ((shv > 255) ? 8'd255 : shv[7:0]);
        wire [7:0] pmax_nx = (sub == 2'd0) ? q : ((q > pmax) ? q : pmax);
        always @(posedge clk) if (state == S_UPD) pmax <= pmax_nx;
        always @(posedge clk) begin
            if (state == S_INIT) begin gsum0 <= 16'd0; gsum1 <= 16'd0; end
            else if (state == S_UPD && layer == 3'd3) begin
                if (g == 1'b0) gsum0 <= gsum0 + {8'd0, q}; else gsum1 <= gsum1 + {8'd0, q};
            end
        end
        assign pmax_flat[l*8 +: 8]  = pmax_nx;
        assign gap0_flat[l*8 +: 8]  = (gsum0 + 16'd32) >> 6;
        assign gap1_flat[l*8 +: 8]  = (gsum1 + 16'd32) >> 6;
        assign t_flat[l*32 +: 32]   = t;
    end endgenerate

    // ------------------------------------------------------------ pooled output write
    wire [3:0] sh_g = {logw, 1'b0} - 4'd2;
    wire [3:0] sh_y = {1'b0, logw} - 4'd1;
    wire [9:0] out_addr = ({9'b0, g} << sh_g) + ({4'b0, py} << sh_y) + {4'b0, px};
    wire wr_pool = (state == S_UPD) && pooled && (sub == 2'd3);
    always @(posedge clk) begin
        if (wr_pool && (layer == 3'd0 || layer == 3'd2)) bufA[out_addr] <= pmax_flat;
        if (wr_pool && layer == 3'd1)                    bufB[out_addr[7:0]] <= pmax_flat;
    end

    assign res_logit = logit_flat[{res_idx, 5'b00000} +: 32];

    // ------------------------------------------------------------ FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0; cycle_count <= 32'd0; layer <= 3'd0; g <= 1'b0;
            py <= 6'd0; px <= 6'd0; sub <= 2'd0; kc <= 10'd0; ic <= 7'd0; ky <= 2'd0; kx <= 2'd0;
            fcin_flat <= 512'd0; logit_flat <= 256'd0;
        end else begin
            if (busy) cycle_count <= cycle_count + 32'd1;
            case (state)
                S_IDLE: if (start) begin
                    busy <= 1'b1; done <= 1'b0; cycle_count <= 32'd0; layer <= 3'd0; g <= 1'b0;
                    py <= 6'd0; px <= 6'd0; sub <= 2'd0; state <= S_INIT;
                end
                S_INIT: state <= S_PREP;
                S_PREP: begin kc <= 10'd0; ic <= 7'd0; ky <= 2'd0; kx <= 2'd0; state <= S_ISSUE; end
                S_ISSUE: begin
                    kc <= kc + 10'd1;
                    if (kc < Kmax && layer != 3'd4) begin
                        if (kx == 2'd2) begin
                            kx <= 2'd0;
                            if (ky == 2'd2) begin ky <= 2'd0; ic <= ic + 7'd1; end else ky <= ky + 2'd1;
                        end else kx <= kx + 2'd1;
                    end
                    if (kc == Kmax) state <= S_R1;
                end
                S_R1: state <= S_R2;
                S_R2: state <= S_R3;
                S_R3: state <= (layer == 3'd4) ? S_FCST : S_UPD;
                S_UPD: begin
                    if (pooled && sub != 2'd3) begin
                        sub <= sub + 2'd1; state <= S_PREP;
                    end else begin
                        sub <= 2'd0;
                        if (px != wlim[5:0] - 6'd1) begin px <= px + 6'd1; state <= S_PREP; end
                        else begin
                            px <= 6'd0;
                            if (py != wlim[5:0] - 6'd1) begin py <= py + 6'd1; state <= S_PREP; end
                            else begin
                                py <= 6'd0;
                                if ({1'b0, g} != ng - 2'd1) begin g <= 1'b1; state <= S_PREP; end
                                else begin
                                    g <= 1'b0;
                                    if (layer == 3'd3) state <= S_GAP;
                                    else begin layer <= layer + 3'd1; state <= S_PREP; end
                                end
                            end
                        end
                    end
                end
                S_GAP: begin
                    fcin_flat <= {gap1_flat, gap0_flat}; layer <= 3'd4; g <= 1'b0;
                    py <= 6'd0; px <= 6'd0; sub <= 2'd0; state <= S_PREP;
                end
                S_FCST: begin
                    logit_flat <= t_flat[255:0];
                    busy <= 1'b0; done <= 1'b1; state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
