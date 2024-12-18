`include "common.vh"

module LEADING_ONE_DETECT #(
    parameter WORD_WID = 8
) (
    input   [WORD_WID-1:0]          d_i,
    output  [$clog2(WORD_WID)-1:0]  idx_o
);

    localparam  IDX_WID         = $clog2(WORD_WID);
    localparam  TREE_LVLS       = $clog2(WORD_WID);             // number of tree levels (leaf node level is excluded)
    localparam  TREE_SIZE       = (1 << (TREE_LVLS + 1)) - 1;   // tree size (leaf node level is added)
    localparam  TREE_LF_OSET    = (1 << TREE_LVLS) - 1;         // tree leaf node offset

    genvar  lvl_gv, idx_gv;

    wire    [IDX_WID-1:0]   idx_tree    [TREE_SIZE];
    wire                    bit_tree    [TREE_SIZE];
    wire                    sel_tree    [(1 << TREE_LVLS)];

    generate
        for (idx_gv = 0; idx_gv < WORD_WID; idx_gv = idx_gv + 1) begin
            assign idx_tree[TREE_LF_OSET + idx_gv] = idx_gv;
            assign bit_tree[TREE_LF_OSET + idx_gv] = d_i[idx_gv];
        end

        for (lvl_gv = 0; lvl_gv < TREE_LVLS; lvl_gv = lvl_gv + 1) begin: TREE_LVL_IT
            for (idx_gv = 0; idx_gv < (1 << lvl_gv); idx_gv = idx_gv + 1) begin: TREE_NODE_IT
                assign sel_tree[`TREE_IDX(lvl_gv, idx_gv)] = bit_tree[`TREE_IDX_PR(lvl_gv, idx_gv)] ? 1 : 0;

                assign idx_tree[`TREE_IDX(lvl_gv, idx_gv)] = (sel_tree[`TREE_IDX(lvl_gv, idx_gv)] == 0) ? idx_tree[`TREE_IDX_PL(lvl_gv, idx_gv)] : 
                                                                                                          idx_tree[`TREE_IDX_PR(lvl_gv, idx_gv)];
                assign bit_tree[`TREE_IDX(lvl_gv, idx_gv)] = (sel_tree[`TREE_IDX(lvl_gv, idx_gv)] == 0) ? bit_tree[`TREE_IDX_PL(lvl_gv, idx_gv)] : 
                                                                                                          bit_tree[`TREE_IDX_PR(lvl_gv, idx_gv)];
            end
        end
    endgenerate
    
    assign idx_o = idx_tree[0];

endmodule