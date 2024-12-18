CC=iverilog
CFLAGS=-g2005-sv -I ./srcs/inc/


test_arbiter:
	$(CC) $(CFLAGS) -o ./bin/BUS_ARBITER ./srcs/common/BUS_ARBITER.sv ./srcs/common/tb/BUS_ARBITER_tb.sv
	vvp ./bin/BUS_ARBITER

test_lod:
	$(CC) $(CFLAGS) -o ./bin/LEADING_ONE_DETECT ./srcs/common/LEADING_ONE_DETECT.sv ./srcs/common/tb/LEADING_ONE_DETECT_tb.sv
	vvp ./bin/LEADING_ONE_DETECT

test_interface:
	$(CC) $(CFLAGS) -o ./bin/SC_CH_SEL_IF_TEST ./srcs/cache/tb/SC_CH_SEL_TEST_MASTER.sv ./srcs/cache/tb/SC_CH_SEL_TEST_SLAVE.sv ./srcs/cache/tb/SC_CH_BANK_IF_TEST_tb.sv
	vvp ./bin/SC_CH_SEL_IF_TEST

test_xbar:
	$(CC) $(CFLAGS) -o ./bin/SC_CH_BANK_XBAR ./srcs/common/BUS_ARBITER.sv ./srcs/common/BUS_QUEUE.sv ./srcs/cache/SC_CH_BANK_XBAR.sv ./srcs/cache/tb/SC_CH_SEL_TEST_MASTER.sv ./srcs/cache/tb/SC_CH_SEL_TEST_SLAVE.sv ./srcs/cache/tb/SC_CH_BANK_XBAR_tb.sv
	vvp ./bin/SC_CH_BANK_XBAR