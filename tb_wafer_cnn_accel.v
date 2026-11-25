`timescale 1ns/1ps
module tb;
    reg clk = 0, rst_n = 0; always #5 clk = ~clk;
    reg img_we = 0; reg [9:0] img_waddr = 0; reg [31:0] img_wdata = 0; reg start = 0;
    wire busy, done; reg [2:0] res_idx = 0; wire [31:0] res_logit; wire [31:0] cycle_count;
    wafer_cnn_accel dut(.clk(clk), .rst_n(rst_n), .img_we(img_we), .img_waddr(img_waddr), .img_wdata(img_wdata),
        .start(start), .busy(busy), .done(done), .res_idx(res_idx), .res_logit(res_logit), .cycle_count(cycle_count));
    reg [31:0] mapmem [0:1023];
    reg [1023:0] mapfile; integer i;
    initial begin
        if (!$value$plusargs("MAP=%s", mapfile)) begin $display("need +MAP=file"); $finish; end
        $readmemh(mapfile, mapmem);
        repeat (4) @(posedge clk); rst_n = 1; repeat (2) @(posedge clk);
        for (i = 0; i < 1024; i = i + 1) begin
            @(posedge clk); img_we <= 1; img_waddr <= i; img_wdata <= mapmem[i];
        end
        @(posedge clk); img_we <= 0;
        @(posedge clk); start <= 1; @(posedge clk); start <= 0;
        wait (done == 1);
        @(posedge clk);
        for (i = 0; i < 8; i = i + 1) begin res_idx = i; #1; $display("LOGIT %0d %0d", i, $signed(res_logit)); end
        $display("CYCLES %0d", cycle_count);
        $finish;
    end
endmodule
