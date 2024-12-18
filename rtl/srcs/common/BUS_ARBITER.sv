`include "common.vh"

module BUS_ARBITER #(
    parameter    CH_NUM    = 8
) (
    clk,        // input:    global positive edge triggered clock
    reset_n,    // input:    global active low reset
    enable,     // input:    enable signal (if enable is not positive, the arbiter maintains its output)
    req_mask,   // input:    request mask (bitmask indicating whether the corresponding client requests for the bus)
    req_idx     // output:    selected client index
);
    /*************************************************************
     * Overview
     *************************************************************
     Description
         *    Arbiter to select a bus request between multiple channels
        *    Round-robin scheduling scheme
        *    Comparator tree based arbiter architecture
     */

    /*************************************************************
     * Local Parameters
     *************************************************************
     Parameter List
        1)  CH_WID:         width of the request index
        2)  TREE_LVLS:      level of the comparator tree
        3)  TREE_SIZE:      number of nodes included in the comparator tree
        4)  TREE_LF_OSET:   offset for the leaf tree nodes
     */

    localparam  CH_WID          = $clog2(CH_NUM);                // width of client index
    localparam  TREE_LVLS       = $clog2(CH_NUM);                // number of tree levels (leaf node level is excluded)
    localparam  TREE_SIZE       = (1 << (TREE_LVLS + 1)) - 1;    // tree size (leaf node level is added)
    localparam  TREE_LF_OSET    = (1 << TREE_LVLS) - 1;         // tree leaf node offset

    /*************************************************************
     * Ports
     *************************************************************
     Description
         *    TBD
     */

    input   wire                    clk;
    input   wire                    reset_n;
    input   wire                    enable;
    input   wire    [CH_NUM-1:0]    req_mask;
    output  reg     [CH_WID-1:0]    req_idx;

    /*************************************************************
     * Comparator Tree Implementation
     *************************************************************
    Comparator Tree based Arbiter
        *    Arbiter select the request with the lowest priority value (lower the higher!)
        *    Binary tree checks wheter the request is issued or not and passes through the request with
                lower priority value
        *    The index of the selected request will be passed to the bus selector

    Variables
        *    prio_r:    registers that store the priority value of each request
                        this register will be initilized from 0 to (CH_NUM-1)
        *    prio_tree: tree for the priority value
        *    req_tree:  tree for the request signal
        *    idx_tree:  tree for the request index
        *    sel_tree:  tree for the selection signal of each node

    Tree Implementation
        *    The tree is an array 
        *    tree[0] is the root node (or the output of the tree)
        *    tree[TREE_LF_OSET:-1] is the leaf node (or the input of the tree)
        *    The input of the tree are the request signals and the priority value
     */

    genvar    lvl_gv, idx_gv;
    integer    idx_it;

    reg        [CH_WID-1:0]    prio_r        [0:CH_NUM-1];
    
    wire    [CH_WID-1:0]    prio_tree    [0:TREE_SIZE-1];
    wire                    req_tree    [0:TREE_SIZE-1];
    wire    [CH_WID-1:0]    idx_tree    [0:TREE_SIZE-1];
    wire                    sel_tree    [0:(1 << TREE_LVLS)-1];

    generate
        for (idx_gv = 0; idx_gv < CH_NUM; idx_gv = idx_gv + 1) begin: TREE_LEAF_NODE
            assign prio_tree[TREE_LF_OSET + idx_gv][CH_WID-1:0]    = prio_r[idx_gv];
            assign req_tree [TREE_LF_OSET + idx_gv]                    = req_mask[idx_gv];
            assign idx_tree [TREE_LF_OSET + idx_gv]                    = idx_gv;
        end

        for (lvl_gv = 0; lvl_gv < TREE_LVLS; lvl_gv = lvl_gv + 1) begin: TREE_LVL_IT
            for (idx_gv = 0; idx_gv < (1 << lvl_gv); idx_gv = idx_gv + 1) begin: TREE_NODE_IT
                assign sel_tree[`TREE_IDX(lvl_gv, idx_gv)] = (req_tree[`TREE_IDX_PL(lvl_gv, idx_gv)] & req_tree[`TREE_IDX_PR(lvl_gv, idx_gv)]) ? `MIN_IDX(prio_tree[`TREE_IDX_PL(lvl_gv, idx_gv)], prio_tree[`TREE_IDX_PR(lvl_gv, idx_gv)])    : 
                                                             req_tree[`TREE_IDX_PR(lvl_gv, idx_gv)] ? 1 : 0;

                assign prio_tree[`TREE_IDX(lvl_gv, idx_gv)] = (sel_tree[`TREE_IDX(lvl_gv, idx_gv)] == 0) ? prio_tree[`TREE_IDX_PL(lvl_gv, idx_gv)] : prio_tree[`TREE_IDX_PR(lvl_gv, idx_gv)];
                assign req_tree [`TREE_IDX(lvl_gv, idx_gv)] = (sel_tree[`TREE_IDX(lvl_gv, idx_gv)] == 0) ? req_tree [`TREE_IDX_PL(lvl_gv, idx_gv)] : req_tree [`TREE_IDX_PR(lvl_gv, idx_gv)];
                assign idx_tree [`TREE_IDX(lvl_gv, idx_gv)] = (sel_tree[`TREE_IDX(lvl_gv, idx_gv)] == 0) ? idx_tree [`TREE_IDX_PL(lvl_gv, idx_gv)] : idx_tree [`TREE_IDX_PR(lvl_gv, idx_gv)];
            end
        end
    endgenerate

    /*************************************************************
     * Output Assignments
     *************************************************************
     Description
         *    TBD
     */

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (idx_it = 0; idx_it < CH_NUM; idx_it = idx_it + 1) begin
                prio_r[idx_it] <= idx_it;
            end
        end else if (enable) begin
            for (idx_it = 0; idx_it < CH_NUM; idx_it = idx_it + 1) begin
                prio_r[idx_it] <= prio_r[idx_it] - prio_tree[0] - 1;
            end
        end
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            req_idx <= 'd0;
        end else if (enable) begin
            req_idx <= idx_tree[0];
        end
    end

endmodule