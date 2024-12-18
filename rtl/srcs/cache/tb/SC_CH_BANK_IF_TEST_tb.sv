module tb_module;

    // Parameters
    parameter CLOCK_PS    = 10000;
    parameter HCLOCK_PS    = 5000;

    parameter BUS_WID    = 8;
    parameter ADDR_WID    = 64;

    // Instanciation
    reg                clk;
    reg                reset_n;
    reg                mode;

    wire                    req;
    wire                    ack;
    wire    [ADDR_WID-1:0]    addr;
    wire                    req_type;
    wire    [BUS_WID-1:0]    rd_data;
    wire    [BUS_WID-1:0]    wr_data;

    SC_CH_SEL_TEST_MASTER #(
        .BUS_WID(BUS_WID), .ADDR_WID(ADDR_WID)
    ) master_mod (
        .clk(clk), .reset_n(reset_n),
        .req_o(req), .ack_i(ack), .addr_o(addr), .req_type_o(req_type), .rd_data_i(rd_data), .wr_data_o(wr_data)
    );
    
    SC_CH_SEL_TEST_SLAVE #(
        .BUS_WID(BUS_WID), .ADDR_WID(ADDR_WID)
    ) slave_mod (
        .clk(clk), .reset_n(reset_n),
        .req_i(req), .ack_o(ack), .addr_i(addr), .req_type_i(req_type), .rd_data_o(rd_data), .wr_data_i(wr_data)
    );

    integer timestamp, i;


    // Clock signal generation
    initial begin : CLOCK_GENERATOR
        clk = 1'b0;
        timestamp = 0;
        forever
            # HCLOCK_PS clk = ~clk;
    end

    always @(posedge clk) begin
        timestamp = timestamp + 1;
    end


    // Test
    initial begin
        $dumpfile("vcd/SC_CH_BANK_IF_TEST_tb.vcd");
        $dumpvars(0);

        reset_n = 1'b0;
        # HCLOCK_PS;
        # HCLOCK_PS;
        reset_n = 1'b1;

        #CLOCK_PS;
        #CLOCK_PS;
        #CLOCK_PS;
        #CLOCK_PS;
        #CLOCK_PS;
        #CLOCK_PS;
        #CLOCK_PS;
        #CLOCK_PS;
        #CLOCK_PS;
        #CLOCK_PS;

        $finish;
    end

    
endmodule
