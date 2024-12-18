module SC_CH_SEL_TEST_MASTER #(
    parameter BUS_WID   = 32,
    parameter ADDR_WID  = 64
) (
    input clk,
    input reset_n,

    output                  req_o,
    output  [ADDR_WID-1:0]  addr_o,
    output                  req_type_o,
    output  [BUS_WID-1:0]   wr_data_o,

    input                   ack_i,
    input   [BUS_WID-1:0]   rd_data_i
);
    reg [BUS_WID-1:0]   data, data_nxt;
    reg [ADDR_WID-1:0]  addr, addr_nxt;
    reg                 req,  req_nxt;

    always @(*) begin
        if (ack_i) begin
            data_nxt = rd_data_i;
            addr_nxt = addr + BUS_WID;
        end else begin
            data_nxt = data;
            addr_nxt = addr;
        end

        req_nxt = 1'b1;
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            data    <= 'd0;
            addr    <= 'd0;
            req     <= 1'b0;
        end else begin
            data    <= data_nxt;
            addr    <= addr_nxt;
            req     <= req_nxt;
        end
    end

    assign req_o        = req && (!ack_i);
    assign addr_o       = addr;
    assign req_type_o   = 1'b0;
    assign wr_data_o    = 'd0;
    
endmodule