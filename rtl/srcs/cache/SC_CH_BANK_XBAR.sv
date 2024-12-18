`include "common.vh"

module SC_CH_BANK_XBAR #(
    parameter   BUS_WID     = 32,
    parameter   ADDR_WID    = 64,
    parameter   CH_NUM      = 8,
    parameter   BANK_NUM    = 8,    // total number of banks
    parameter   BANK_OSET   = 5
) (
    input    clk,
    input    reset_n,
    input    mode,

    input                   ch_req_i        [CH_NUM],
    input   [ADDR_WID-1:0]  ch_addr_i       [CH_NUM],
    input                   ch_req_type_i   [CH_NUM],
    input   [BUS_WID-1:0]   ch_wr_data_i    [CH_NUM],
    output                  ch_ack_o        [CH_NUM],
    output  [BUS_WID-1:0]   ch_rd_data_o    [CH_NUM],

    output                  bank_req_o      [BANK_NUM],
    output  [ADDR_WID-1:0]  bank_addr_o     [BANK_NUM],
    output                  bank_req_type_o [BANK_NUM],
    output  [BUS_WID-1:0]   bank_wr_data_o  [BANK_NUM],
    input                   bank_ack_i      [BANK_NUM],
    input   [BUS_WID-1:0]   bank_rd_data_i  [BANK_NUM]
);
    localparam    CH_WID    = $clog2(CH_NUM);
    localparam    BANK_WID  = $clog2(BANK_NUM);
    
    localparam    CH_BANK_MAP_TREE_LVLS     = $clog2(BANK_NUM);
    localparam    CH_BANK_MAP_TREE_SIZE     = (1 << (CH_BANK_MAP_TREE_LVLS + 1)) - 1;
    localparam    CH_BANK_MAP_TREE_LF_OSET  = (1 << CH_BANK_MAP_TREE_LVLS) - 1;

    genvar ch_gv, bank_gv, lvl_gv;
    integer ch_it;

    wire    [BANK_WID-1:0]  ch_bank_idx     [CH_NUM];

    generate
        for (ch_gv = 0; ch_gv < CH_NUM; ch_gv = ch_gv + 1) begin
            assign ch_bank_idx[ch_gv] = ch_addr_i[ch_gv][BANK_OSET+:BANK_WID];
        end
    endgenerate

    reg                     arb_enable      [BANK_NUM];
    wire    [CH_NUM-1:0]    arb_req_mask    [BANK_NUM];  // request mask 
    wire    [CH_WID-1:0]    arb_ch_idx      [BANK_NUM];

    generate
        for (bank_gv = 0; bank_gv < BANK_NUM; bank_gv = bank_gv + 1) begin
            BUS_ARBITER #(
                .CH_NUM(CH_NUM)
            ) arbiter (
                .clk        (clk), 
                .reset_n    (reset_n),
                .enable     (arb_enable[bank_gv]), 
                .req_mask   (arb_req_mask[bank_gv]), 
                .req_idx    (arb_ch_idx[bank_gv])
            );

            for (ch_gv = 0; ch_gv < CH_NUM; ch_gv = ch_gv + 1) begin
                assign arb_req_mask[bank_gv][ch_gv] = (ch_bank_idx[ch_gv] == bank_gv) && ch_req_i[ch_gv];
            end

            assign arb_enable[bank_gv] = bank_ack_i[bank_gv];
        end
    endgenerate

    generate
        // channel to bank
        for (bank_gv = 0; bank_gv < BANK_NUM; bank_gv = bank_gv + 1) begin
            assign bank_req_o      [bank_gv] = ch_req_i      [arb_ch_idx[bank_gv]] & arb_req_mask[bank_gv];
            assign bank_addr_o     [bank_gv] = ch_addr_i     [arb_ch_idx[bank_gv]];
            assign bank_req_type_o [bank_gv] = ch_req_type_i [arb_ch_idx[bank_gv]];
            assign bank_wr_data_o  [bank_gv] = ch_wr_data_i  [arb_ch_idx[bank_gv]];
        end

        // bank to channel
        for (ch_gv = 0; ch_gv < CH_NUM; ch_gv = ch_gv + 1) begin
            assign ch_ack_o     [ch_gv] = bank_ack_i     [ch_bank_idx[ch_gv]] & (arb_ch_idx[ch_bank_idx[ch_gv]] == ch_gv);
            assign ch_rd_data_o [ch_gv] = bank_rd_data_i [ch_bank_idx[ch_gv]];
        end
    endgenerate
    
endmodule