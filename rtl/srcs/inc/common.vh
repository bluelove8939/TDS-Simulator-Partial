`ifndef __COMMON_VH
`define __COMMON_VH

// MACROS:  BUS_ARBITER
`define TREE_IDX(lvl, idx)      (((1 << (lvl)) - 1) + (idx))
`define TREE_IDX_PL(lvl, idx)   ((`TREE_IDX((lvl), (idx)) << 1) + 1)
`define TREE_IDX_PR(lvl, idx)   ((`TREE_IDX((lvl), (idx)) << 1) + 2)

`define MIN_IDX(x, y)   (((x) < (y)) ? (0) : (1))

// MACROS:  CACHE ADDRESS PARSING
`define PARSE_

`endif