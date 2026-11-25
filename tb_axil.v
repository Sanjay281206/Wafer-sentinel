`timescale 1ns/1ps
module tb_axil;
    reg clk = 0, rstn = 0; always #5 clk = ~clk;
    reg [12:0] awaddr = 0, araddr = 0; reg awvalid = 0, wvalid = 0, bready = 1, arvalid = 0, rready = 1; reg [31:0] wdata = 0;
    wire awready, wready, bvalid, arready, rvalid; wire [31:0] rdata; wire [1:0] bresp, rresp;
    wafer_cnn_axil dut(.s_axi_aclk(clk), .s_axi_aresetn(rstn), .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid),
        .s_axi_awready(awready), .s_axi_wdata(wdata), .s_axi_wstrb(4'hF), .s_axi_wvalid(wvalid), .s_axi_wready(wready),
        .s_axi_bresp(bresp), .s_axi_bvalid(bvalid), .s_axi_bready(bready), .s_axi_araddr(araddr),
        .s_axi_arvalid(arvalid), .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid), .s_axi_rready(rready));
    task axi_w(input [12:0] a, input [31:0] d); begin
        @(posedge clk); awaddr <= a; wdata <= d; awvalid <= 1; wvalid <= 1;
        wait (awready && wready); @(posedge clk); awvalid <= 0; wvalid <= 0; wait (bvalid); @(posedge clk);
    end endtask
    task axi_r(input [12:0] a, output [31:0] d); begin
        @(posedge clk); araddr <= a; arvalid <= 1; wait (arready); @(posedge clk); arvalid <= 0;
        wait (rvalid); d = rdata; @(posedge clk);
    end endtask
    reg [31:0] mapmem [0:1023]; reg [1023:0] mapfile; reg [31:0] v; integer i;
    initial begin
        if (!$value$plusargs("MAP=%s", mapfile)) $finish;
        $readmemh(mapfile, mapmem);
        repeat (5) @(posedge clk); rstn = 1; repeat (3) @(posedge clk);
        for (i = 0; i < 1024; i = i + 1) axi_w(13'h1000 + 4*i, mapmem[i]);
        axi_w(13'h0000, 32'd1);
        v = 0; while (!v[1]) begin axi_r(13'h0004, v); end
        axi_r(13'h0008, v); $display("CYCLES %0d", v);
        for (i = 0; i < 8; i = i + 1) begin axi_r(13'h0010 + 4*i, v); $display("LOGIT %0d %0d", i, $signed(v)); end
        axi_r(13'h0004, v); $display("STATUS %b", v[1:0]);
        $finish;
    end
endmodule
