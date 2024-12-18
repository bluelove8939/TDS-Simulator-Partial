module SC_CH_SEL_TEST_SLAVE #(
    parameter BUS_WID   = 32,
    parameter ADDR_WID  = 64
) (
    input clk,
    input reset_n,

    input                   req_i,
    input   [ADDR_WID-1:0]  addr_i,
    input                   req_type_i,
    input   [BUS_WID-1:0]   wr_data_i,

    output                  ack_o,
    output  [BUS_WID-1:0]   rd_data_o
);
    localparam BUS_ADDR_FIELD_WID = $clog2(BUS_WID);
    localparam DELAY = 3;

    reg [1:0] cnt;

    reg                 ack, ack_nxt;
    reg [BUS_WID-1:0]   rd_data, rd_data_nxt;

    reg [BUS_WID-1:0]   mem [1024];
    integer i;    

    initial begin
        for (i = 0; i < 128; i = i + 1) begin
            mem[i] = i+1;
        end
    end

    always @(*) begin
        if (req_i) begin
            rd_data_nxt = mem[addr_i >> BUS_ADDR_FIELD_WID];
            ack_nxt     = 1'b1;
        end else begin
            rd_data_nxt = rd_data;
            ack_nxt     = 1'b0;
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            rd_data <= 'd0;
            ack     <= 1'b0;
        end else begin
            rd_data <= rd_data_nxt;
            ack     <= ack_nxt;
        end
    end

    assign ack_o        = ack;
    assign rd_data_o    = rd_data;
    
endmodule