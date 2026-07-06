current_design alu_4bit

set clk_period 6.5
create_clock -name virtual_clk -period $clk_period

set_input_delay 2.0 \
    -clock virtual_clk \
    [all_inputs]

set_output_delay 2.0 \
    -clock virtual_clk \
    [all_outputs]
