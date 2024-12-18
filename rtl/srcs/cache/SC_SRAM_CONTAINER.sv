module SC_SRAM_CONTAINER #(
    parameter   WORD_WID    = 8,
    parameter   ADDR_WID    = 32,
) (
    input                   clk,
    input                   reset_n,

    input                   wr_enable_i,
    input   [ADDR_WID-1:0]  addr_i,
    input   [WORD_WID-1:0]  wr_data_i,
    output  [WORD_WID-1:0]  rd_data_o

);

// sysnopsys translate_off

    localparam  ENT_NUM = 1 << ADDR_WID;

    integer i;

    reg [WORD_WID-1:0]  container   [ENT_NUM];

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (i = 0; i < ENT_NUM; i = i + 1) begin
                container[i] <= 'd0;
            end
        end else if (wr_enable_i) begin
            container[i] <= wr_data_i;
        end
    end

    assign rd_data_o = (wr_enable_i == 1'b0) ? container[addr_i] : 'd0;

// sysnopsys translate_on
    
endmodule