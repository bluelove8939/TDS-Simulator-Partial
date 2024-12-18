module tb_module;

    // Parameters
    parameter   CLOCK_PS    = 10000;
    parameter   HCLOCK_PS   = 5000;
    parameter   WORD_WID    = 8;
    parameter   IDX_WID     = $clog2(WORD_WID);

    // Instanciation
    reg     [WORD_WID-1:0]  d_i;
    wire    [IDX_WID-1:0]   idx_o;

    LEADING_ONE_DETECT #(
        .WORD_WID(WORD_WID)
    ) top (
        .d_i(d_i), .idx_o(idx_o)
    );

    // Test
    initial begin
        $dumpfile("vcd/LEADING_ONE_DETECT_tb.vcd");
        $dumpvars(0);

        for (d_i = 0; d_i < ((1 << WORD_WID) - 1); d_i = d_i + 1) begin
            # CLOCK_PS;
            $display("d_i: %8b  idx_o: %1d", d_i, idx_o);
        end

        $finish;
    end

    
endmodule
