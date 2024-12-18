module tb_module;

    // Parameters
    parameter  CLOCK_PS = 10000;
    parameter  HCLOCK_PS = 5000;


    // Instanciation
    reg                clk;
    reg                reset_n;
    reg                enable;
    reg        [7:0]    req_mask;
    wire    [2:0]    req_idx;

    BUS_ARBITER #(
        .CH_NUM(8)
    ) top (
        .clk(clk), .reset_n(reset_n),
        .enable(enable), .req_mask(req_mask), .req_idx(req_idx)
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
        $dumpfile("vcd/BUS_ARBITER_tb.vcd");
        $dumpvars(0);

        enable = 1'b0;
        req_mask = 8'b0000_0000;

        reset_n = 1'b0;
        # HCLOCK_PS;
        # HCLOCK_PS;
        reset_n = 1'b1;

        enable = 1'b0;
        req_mask = 8'b1001_0010;

        for (i = 0; i < 8; i = i + 1) begin
            #CLOCK_PS;
            $display("#%3d   enable: %b   request mask: %b   selected index: %d", timestamp, enable, req_mask, req_idx);
        end

        enable = 1'b1;
        req_mask = 8'b1001_0010;

        for (i = 0; i < 8; i = i + 1) begin
            // #CLOCK_PS;
            $display("#%3d   enable: %b   request mask: %b   selected index: %d", timestamp, enable, req_mask, req_idx);
            #CLOCK_PS;
        end

        enable = 1'b1;
        req_mask = 8'b1111_1111;

        for (i = 0; i < 5; i = i + 1) begin
            // #CLOCK_PS;
            $display("#%3d   enable: %b   request mask: %b   selected index: %d", timestamp, enable, req_mask, req_idx);
            #CLOCK_PS;
        end

        enable = 1'b0;
        for (i = 0; i < 3; i = i + 1) begin
            // #CLOCK_PS;
            $display("#%3d   enable: %b   request mask: %b   selected index: %d", timestamp, enable, req_mask, req_idx);
            #CLOCK_PS;
        end

        $finish;
    end

    
endmodule
