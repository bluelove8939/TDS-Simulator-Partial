module tb_module;

    // Parameters
    parameter   CLOCK_PS    = 10000;
    parameter   HCLOCK_PS   = 5000;

    parameter   BUS_WID     = 8;
    parameter   ADDR_WID    = 64;
    parameter   CH_NUM      = 2;
    parameter   BANK_NUM    = 2;    // total number of banks
    parameter   BANK_OSET   = 3;

    // Instanciation
    reg                clk;
    reg                reset_n;
    reg                mode;

    wire                    ch_req        [CH_NUM];
    wire                    ch_ack        [CH_NUM];
    wire    [ADDR_WID-1:0]    ch_addr        [CH_NUM];
    wire                    ch_req_type    [CH_NUM];
    wire    [BUS_WID-1:0]    ch_rd_data    [CH_NUM];
    wire    [BUS_WID-1:0]    ch_wr_data    [CH_NUM];

    wire                    bank_req        [BANK_NUM];
    wire                    bank_ack        [BANK_NUM];
    wire    [ADDR_WID-1:0]    bank_addr        [BANK_NUM];
    wire                    bank_req_type    [BANK_NUM];
    wire    [BUS_WID-1:0]    bank_rd_data    [BANK_NUM];
    wire    [BUS_WID-1:0]    bank_wr_data    [BANK_NUM];

    genvar ch_gv, bank_gv;

    generate
        for (ch_gv = 0; ch_gv < CH_NUM; ch_gv = ch_gv + 1) begin    
            SC_CH_SEL_TEST_MASTER #(
                .BUS_WID    (BUS_WID), 
                .ADDR_WID   (ADDR_WID)
            ) channel_mod (
                .clk        (clk), 
                .reset_n    (reset_n),
                .req_o      (ch_req[ch_gv]), 
                .ack_i      (ch_ack[ch_gv]), 
                .addr_o     (ch_addr[ch_gv]), 
                .req_type_o (ch_req_type[ch_gv]), 
                .rd_data_i  (ch_rd_data[ch_gv]), 
                .wr_data_o  (ch_wr_data[ch_gv])
            );
        end

        for (bank_gv = 0; bank_gv < BANK_NUM; bank_gv = bank_gv + 1) begin
            SC_CH_SEL_TEST_SLAVE #(
                .BUS_WID    (BUS_WID), 
                .ADDR_WID   (ADDR_WID)
            ) bank_mod (
                .clk        (clk), 
                .reset_n    (reset_n),
                .req_i      (bank_req[bank_gv]), 
                .ack_o      (bank_ack[bank_gv]), 
                .addr_i     (bank_addr[bank_gv]), 
                .req_type_i (bank_req_type[bank_gv]), 
                .rd_data_o  (bank_rd_data[bank_gv]), 
                .wr_data_i  (bank_wr_data[bank_gv])
            );
        end
    endgenerate

    SC_CH_BANK_XBAR #(
        .BUS_WID            (BUS_WID),
        .ADDR_WID            (ADDR_WID),
        .CH_NUM                (CH_NUM),
        .BANK_NUM            (BANK_NUM),
        .BANK_OSET            (BANK_OSET)
    ) xbar_mod (
        .clk                (clk), 
        .reset_n            (reset_n), 
        .mode                (mode),
        
        .ch_req_i            (ch_req), 
        .ch_ack_o            (ch_ack), 
        .ch_addr_i            (ch_addr), 
        .ch_req_type_i        (ch_req_type), 
        .ch_rd_data_o        (ch_rd_data), 
        .ch_wr_data_i        (ch_wr_data),

        .bank_req_o            (bank_req), 
        .bank_ack_i            (bank_ack), 
        .bank_addr_o        (bank_addr), 
        .bank_req_type_o    (bank_req_type), 
        .bank_rd_data_i        (bank_rd_data), 
        .bank_wr_data_o        (bank_wr_data)
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
        $dumpfile("vcd/SC_CH_BANK_XBAR_tb.vcd");
        $dumpvars(0);

        reset_n = 1'b0;
        # HCLOCK_PS;
        # HCLOCK_PS;
        reset_n = 1'b1;

        for (i = 0; i < 128; i = i + 1) begin
            # CLOCK_PS;
            $display("[#%3d] arb_ch_idx[0]: %1d [1]: %1d  |  ch_bank_idx[0]: %1d [1]: %1d  |  ch_addr_i[0]: %3d [1]: %3d  |  ch_rd_data_o[0]: %3d [1]: %3d", 
                timestamp,  
                
                xbar_mod.arb_ch_idx[0],
                xbar_mod.arb_ch_idx[1],

                xbar_mod.ch_bank_idx[0],
                xbar_mod.ch_bank_idx[1],

                xbar_mod.ch_addr_i[0],
                xbar_mod.ch_addr_i[1],

                xbar_mod.ch_rd_data_o[0],
                xbar_mod.ch_rd_data_o[1],
            );
        end

        $finish;
    end
endmodule
