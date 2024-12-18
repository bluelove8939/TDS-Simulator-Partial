module GA_CORE #(
    parameter   PE_NUM      = 8,
    parameter   MAC_PER_PE  = 32,
    parameter   WORD_WID    = 8,
    parameter   PTR_WID     = 32,
    parameter   ACC_WID     = 32,
    parameter   DIM_WID     = 32
) (
    clk,
    reset_n,

    // Control IF
    mode_i,
    enable_i,
    response_o,
    dat_a_addr_i,
    dat_b_addr_i,
    idx_a_addr_i,
    ptr_a_addr_i,
    dim_M_size_i,
    dim_N_size_i,
    dim_K_size_i,

    // Cache Channel IF (MASTER)
    ch_req_o,
    ch_addr_o,
    ch_req_type_o,
    ch_wr_data_o,
    ch_ack_i,
    ch_rd_data_i,
    
    // Memory Channel IF (MASTER)
    mem_req_o,
    mem_addr_o,
    mem_req_type_o,
    mem_wr_data_o,
    mem_ack_i,
    mem_rd_data_i
);
    /************************************************************
     * Local Parameters
     ************************************************************/
    
    localparam  CLINE_WID   = PE_NUM * MAC_PER_PE * WORD_WID;

    localparam  IDLE    =   0;
    localparam  GEMM_S0 =   1;
    localparam  GEMM_S1 =   2;
    localparam  GEMM_S2 =   3;
    localparam  GEMM_S3 =   4;
    localparam  SDMM_S0 =   5;
    localparam  SDMM_S1 =   6;
    localparam  SDMM_S2 =   7;
    localparam  SDMM_S3 =   8;
    localparam  SDMM_S4 =   9;
    localparam  SDMM_S5 =   10;

    /************************************************************
     * Port Declaration
     ************************************************************/

    input   clk,
    input   reset_n,

    // Control IF
    input                   mode_i,
    input                   enable_i,
    output                  response_o,
    input   [ADDR_WID-1:0]  dat_a_addr_i,
    input   [ADDR_WID-1:0]  dat_b_addr_i,
    input   [ADDR_WID-1:0]  idx_a_addr_i,
    input   [ADDR_WID-1:0]  ptr_a_addr_i,
    input   [DIM_WID-1:0]   dim_M_size_i,
    input   [DIM_WID-1:0]   dim_N_size_i,
    input   [DIM_WID-1:0]   dim_K_size_i,

    // Cache Channel IF (MASTER)
    output                  ch_req_o        [PE_NUM],
    output  [ADDR_WID-1:0]  ch_addr_o       [PE_NUM],
    output                  ch_req_type_o   [PE_NUM],
    output  [CLINE_WID-1:0] ch_wr_data_o    [PE_NUM],
    input                   ch_ack_i        [PE_NUM],
    input   [CLINE_WID-1:0] ch_rd_data_i    [PE_NUM],

    // Memory Channel IF (MASTER)
    output                  mem_req_o,
    output  [ADDR_WID-1:0]  mem_addr_o,
    output                  mem_req_type_o,
    output  [CLINE_WID-1:0] mem_wr_data_o,
    input                   mem_ack_i,
    input   [CLINE_WID-1:0] mem_rd_data_i

    /************************************************************
     * Output Assignment
     ************************************************************/

    integer pe_it, mac_it;
    genvar  pe_gv, mac_gv;

    // output assignments
    reg     response;

    reg                     ch_req          [PE_NUM], ch_req_nxt        [PE_NUM];
    reg     [ADDR_WID-1:0]  ch_addr         [PE_NUM], ch_addr_nxt       [PE_NUM];
    reg                     ch_req_type     [PE_NUM], ch_req_type_nxt   [PE_NUM];
    reg     [CLINE_WID-1:0] ch_wr_data      [PE_NUM], ch_wr_data_nxt    [PE_NUM];

    reg                     mem_req,        mem_req_nxt;
    reg     [ADDR_WID-1:0]  mem_addr,       mem_addr_nxt;
    reg                     mem_req_type,   mem_req_type_nxt;
    reg     [CLINE_WID-1:0] mem_wr_data,    mem_wr_data_nxt;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
                ch_req[pe_it]       <= 'd0;
                ch_addr[pe_it]      <= 'd0;
                ch_req_type[pe_it]  <= 'd0;
                ch_wr_data[pe_it]   <= 'd0;
            end

            mem_req       <= 'd0;
            mem_addr      <= 'd0;
            mem_req_type  <= 'd0;
            mem_wr_data   <= 'd0;
        end else begin
            for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
                ch_req[pe_it]       <= ch_req_nxt[pe_it];
                ch_addr[pe_it]      <= ch_addr_nxt[pe_it];
                ch_req_type[pe_it]  <= ch_req_type_nxt[pe_it];
                ch_wr_data[pe_it]   <= ch_wr_data_nxt[pe_it];
            end

            mem_req       <= mem_req_nxt;
            mem_addr      <= mem_addr_nxt;
            mem_req_type  <= mem_req_type_nxt;
            mem_wr_data   <= mem_wr_data_nxt;
        end
            
    end

    assign response_o       = response;
    generate
        for (pe_gv = 0; pe_gv < PE_NUM; pe_gv = pe_gv + 1) begin
            assign ch_req_o[pe_it]         = ch_req[pe_it] && (!ch_ack_i[pe_it]);
            assign ch_addr_o[pe_it]        = ch_addr[pe_it];
            assign ch_req_type_o[pe_it]    = ch_req_type[pe_it];
            assign ch_wr_data_o[pe_it]     = ch_wr_data[pe_it];
        end
    endgenerate
    assign mem_req_o        = mem_req && (!mem_ack_i);
    assign mem_addr_o       = mem_addr;
    assign mem_req_type_o   = mem_req_type;
    assign mem_wr_data_o    = mem_wr_data;

    /************************************************************
     * Finite State Machine
     ************************************************************/

    reg     [3:0]   state, state_nxt;

    reg     [ACC_WID-1:0]   acc_registers       [PE_NUM][MAC_PER_PE],
                            acc_registers_nxt   [PE_NUM][MAC_PER_PE];
    reg     [ACC_WID-1:0]   ptrA_buffer         [PE_NUM+1],
                            ptrA_buffer_nxt     [PE_NUM+1];
    reg     [ACC_WID-1:0]   idxA_buffer         [PE_NUM],
                            idxA_buffer_nxt     [PE_NUM];
    reg     [WORD_WID-1:0]  rowA_buffer         [PE_NUM],
                            rowA_buffer_nxt     [PE_NUM];
    reg     [31:0]          rowA_cursor,
                            rowA_cursor_nxt;

    always @(*) begin
        state_nxt = state;

        for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
            ch_req_nxt[pe_it]       = ch_req[pe_it];
            ch_addr_nxt[pe_it]      = ch_addr[pe_it];
            ch_req_type_nxt[pe_it]  = ch_req_type[pe_it];
            ch_wr_data_nxt[pe_it]   = ch_wr_data[pe_it];
        end

        mem_req_nxt       = mem_req;
        mem_addr_nxt      = mem_addr;
        mem_req_type_nxt  = mem_req_type;
        mem_wr_data_nxt   = mem_wr_data;

        for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
            for (mac_it = 0; mac_it < MAC_PER_PE; mac_it = mac_it + 1) begin
                acc_registers_nxt[pe_it][mac_it] = acc_registers[pe_it][mac_it]; 
            end
        end

        for (pe_it = 0; pe_it < (PE_NUM+1); pe_it = pe_it + 1) begin
            ptrA_buffer_nxt[pe_it]  = ptrA_buffer[pe_it];
        end

        for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
            idxA_buffer_nxt[pe_it]  = idxA_buffer[pe_it];
            rowA_buffer_nxt[pe_it]  = rowA_buffer[pe_it];
        end

        rowA_cursor_nxt = rowA_cursor;

        response = 1'b0;

        case (state)
            IDLE: begin
                if (enable_i) begin
                    if (mode_i == 1'b0) begin
                        rowA_cursor_nxt = 'd0;
                        
                        for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
                            for (mac_it = 0; mac_it < MAC_PER_PE; mac_it = mac_it + 1) begin
                                acc_registers_nxt[pe_it][mac_it] = 'd0; 
                            end
                        end

                        state_nxt = GEMM_S0;
                    end else begin
                        state_nxt = SDMM_S0;
                    end
                end

                response = 1'b1;
            end 

            GEMM_S0: begin
                if (rowA_cursor >= dim_K_size) begin
                    state_nxt = IDLE;
                end else begin
                    for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
                        ch_req_nxt[pe_it] = 1'b1;
                        ch_addr_nxt[pe_it] = 
                    end 
                end
            end

            default: 
        endcase
    end

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            state <= IDLE;

            for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
                for (mac_it = 0; mac_it < MAC_PER_PE; mac_it = mac_it + 1) begin
                    acc_registers[pe_it][mac_it] <= 'd0; 
                end
            end

            for (pe_it = 0; pe_it < (PE_NUM+1); pe_it = pe_it + 1) begin
                ptrA_buffer[pe_it]  <= 'd0;
            end

            for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
                idxA_buffer[pe_it]  <= 'd0;
                rowA_buffer[pe_it]  <= 'd0;
            end

            rowA_cursor <= 'd0;
        end else begin
            state <= state_nxt;

            for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
                for (mac_it = 0; mac_it < MAC_PER_PE; mac_it = mac_it + 1) begin
                    acc_registers[pe_it][mac_it] <= acc_registers_nxt[pe_it][mac_it]; 
                end
            end

            for (pe_it = 0; pe_it < (PE_NUM+1); pe_it = pe_it + 1) begin
                ptrA_buffer[pe_it]  <= ptrA_buffer_nxt[pe_it];
            end

            for (pe_it = 0; pe_it < PE_NUM; pe_it = pe_it + 1) begin
                idxA_buffer[pe_it]  <= idxA_buffer_nxt[pe_it];
                rowA_buffer[pe_it]  <= rowA_buffer_nxt[pe_it];
            end

            rowA_cursor <= rowA_cursor_nxt;
        end
    end

endmodule