module SC_BANK #(
    parameter   ADDR_WID    = 64,
    parameter   CLINE_WID   = 32,  // same as the BUS_WID in TEST_SLAVE module
    parameter   SET_NUM     = 32,
    parameter   WAY_NUM     = 8
) (
    input clk,
    input reset_n,

    // Channel IF (SLAVE)
    input                   ch_req_i,
    input   [ADDR_WID-1:0]  ch_addr_i,
    input                   ch_req_type_i,
    input   [CLINE_WID-1:0] ch_wr_data_i,

    output                  ch_ack_o,
    output  [CLINE_WID-1:0] ch_rd_data_o,

    // Memory IF (MASTER)
    output                  mem_req_o,
    output  [ADDR_WID-1:0]  mem_addr_o,
    output                  mem_req_type_o,
    output  [CLINE_WID-1:0] mem_wr_data_o,

    input                   mem_ack_i,
    input   [CLINE_WID-1:0] mem_rd_data_i
);
    
    localparam  CLINE_SIZ   =   CLINE_WID >> 3;
    localparam  BST_FWID    =   $clog2(CLINE_SIZ);
    localparam  SET_FWID    =   $clog2(SET_NUM);
    localparam  WAY_FWID    =   $clog2(WAY_NUM);
    localparam  TAG_FWID    =   ADDR_WID - BST_FWID - SET_FWID;
    localparam  BST_FOST    =   0;
    localparam  SET_FOST    =   BST_FOST + BST_FWID;
    localparam  TAG_FOST    =   SET_FOST + SET_FWID;
    
    localparam  LRU_CNT_WID =   $clog2(SET_NUM);
    localparam  TAG_ENT_WID =   TAG_FWID + 3 + LRU_CNT_WID;
    localparam  LRU_CNT_OST =   0;
    localparam  VALID_B_OST =   LRU_CNT_OST + 3;
    localparam  DIRTY_B_OST =   VALID_B_OST + 1;
    localparam  VICT_B_OST  =   DIRTY_B_OST + 1;
    localparam  TAG_B_OST   =   VICT_B_OST  + 1;

    genvar way_gv;

    // cache entries
    reg                     dat_ent_wr_enable   [WAY_NUM];
    reg     [SET_FWID-1:0]  dat_ent_set_idx     [WAY_NUM];
    reg     [WAY_FWID-1:0]  dat_ent_way_idx     [WAY_NUM];
    reg     [CLINE_WID-1:0] dat_ent_wr_data     [WAY_NUM];
    wire    [CLINE_WID-1:0] dat_ent_rd_data     [WAY_NUM];

    reg                     tag_ent_wr_enable   [WAY_NUM];
    reg     [SET_FWID-1:0]  tag_ent_set_idx     [WAY_NUM];
    reg     [WAY_FWID-1:0]  tag_ent_way_idx     [WAY_NUM];
    reg     [TAG_FWID-1:0]  tag_ent_wr_data     [WAY_NUM];
    wire    [TAG_FWID-1:0]  tag_ent_rd_data     [WAY_NUM];

    generate
        for (way_gv = 0; way_gv < WAY_NUM; way_gv = way_gv + 1) begin
            SC_SRAM_CONTAINER #(
                .WORD_WID   (CLINE_WID),
                .SET_NUM    (SET_NUM),
                .WAY_NUM    (WAY_NUM),
            ) data_container (
                .clk            (clk), 
                .reset_n        (reset_n),
                .wr_enable_i    (dat_ent_wr_enable[way_gv]),
                .set_idx_i      (dat_ent_set_idx[way_gv]),
                .way_idx_i      (dat_ent_way_idx[way_gv]),
                .wr_data_i      (dat_ent_wr_data[way_gv]),
                .rd_data_o      (dat_ent_rd_data[way_gv]),
            );

            SC_SRAM_CONTAINER #(
                .WORD_WID   (TAG_FWID),
                .SET_NUM    (SET_NUM),
                .WAY_NUM    (WAY_NUM),
            ) tag_container (
                .clk            (clk), 
                .reset_n        (reset_n),
                .wr_enable_i    (tag_ent_wr_enable[way_gv]),
                .set_idx_i      (tag_ent_set_idx[way_gv]),
                .way_idx_i      (tag_ent_way_idx[way_gv]),
                .wr_data_i      (tag_ent_wr_data[way_gv]),
                .rd_data_o      (tag_ent_rd_data[way_gv]),
            );
        end
    endgenerate

    // tag matching
    wire    [TAG_FWID-1:0]      parsed_tag;
    wire    [SET_FWID-1:0]      parsed_set;
    wire    [BST_FWID-1:0]      parsed_bst;

    wire    [WAY_NUM-1:0]       tag_mask;       // bitmask indicating tag entry matching result
    wire    [WAY_NUM-1:0]       valid_mask;     // bitmask indicating the tag entry is valid
    wire    [WAY_NUM-1:0]       victim_n_mask;  // bitmask indicating the tag entry is not selected as a victim
    wire    [WAY_NUM-1:0]       sel_way_mask;

    wire    [WAY_FWID-1:0]      sel_way_idx;  // selected way index (created by LOD)

    wire    [TAG_FWID-1:0]      sel_way_tag;
    wire    [CLINE_WID-1:0]     sel_way_data;
    wire    [LRU_CNT_WID-1:0]   sel_way_lru_cnt;
    
    wire                        is_hit;
    wire                        is_valid;
    wire                        is_dirty;
    wire                        is_victim;

    assign parsed_tag   = ch_addr_i[TAG_FOST+:TAG_FWID];
    assign parsed_set   = ch_addr_i[SET_FOST+:SET_FWID];
    assign parsed_bst   = ch_addr_i[BST_FOST+:BST_FWID];

    generate
        for (way_gv = 0; way_gv < WAY_NUM; way_gv = way_gv + 1) begin
            assign tag_mask      [way_gv] =  tag_ent_rd_data[way_gv][TAG_B_OST+:TAG_FWID] == parsed_tag;
            assign valid_mask    [way_gv] =  (tag_ent_rd_data[way_gv][VALID_B_OST+:1]) & (tag_ent_wr_enable[way_gv] == 0);  // if wr_enable is triggered for the way, the tag entry read from the container must be invalid
            assign victim_n_mask [way_gv] = ~tag_ent_rd_data[way_gv][VICT_B_OST+:1];  // invert victim bit and store it to the mask
        end

        assign sel_way_mask = tag_mask & valid_mask & victim_n_mask;
    endgenerate 

    LEADING_ONE_DETECT #(
        .WORD_WID   (WAY_NUM)
    ) sel_way_lod (
        .d_i        (sel_way_mask),
        .idx_o      (sel_way_idx)
    );

    assign sel_way_tag      = tag_ent_rd_data[sel_way_idx][TAG_B_OST+:TAG_FWID];
    assign sel_way_data     = dat_ent_rd_data[sel_way_idx];
    assign sel_way_lru_cnt  = tag_ent_rd_data[sel_way_idx][LRU_CNT_OST+:LRU_CNT_WID];

    assign is_hit       = sel_way_mask[sel_way_idx];
    assign is_valid     = tag_ent_rd_data[sel_way_idx][VALID_B_OST+:1];
    assign is_dirty     = tag_ent_rd_data[sel_way_idx][DIRTY_B_OST+:1];
    assign is_victim    = tag_ent_rd_data[sel_way_idx][VICT_B_OST+:1];

    // channel slave interface
    reg                     ch_ack,     ch_ack_nxt;
    reg     [CLINE_WID-1:0] ch_rd_data, ch_rd_data_nxt;

    always @(*) begin
        if (req_i) begin
            ch_ack_nxt      = is_hit;
            ch_rd_data_nxt  = sel_way_data;
        end else begin
            ch_ack_nxt      = 1'b0;
            ch_rd_data_nxt  = ch_rd_data;
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            ch_ack      <= 1'b0;
            ch_rd_data  <= 'd0;
        end else begin
            ch_ack      <= ch_ack_nxt;
            ch_rd_data  <= ch_rd_data_nxt;
        end
    end

    assign ch_ack_o     = ch_ack;
    assign ch_rd_data_o = ch_rd_data;

endmodule