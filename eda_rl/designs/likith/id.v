module id(

    input  [2:0] opcode,

    output reg rf_write_en,
    output reg display_en,
    output reg alu_enable,
    output reg mux_sel,
    output reg [1:0] alu_op

);

always @(*) begin

    rf_write_en = 0;
    display_en  = 0;
    alu_enable  = 0;
    mux_sel     = 0;
    alu_op      = 2'b00;

    case(opcode)

        3'b000: begin
            // NOP
        end

        3'b001: begin
            // LD
            rf_write_en = 1;
        end

        3'b010: begin
            // MOV
            rf_write_en = 1;
        end

        3'b011: begin
            // DISP
            display_en = 1;
            mux_sel    = 1;
        end

        3'b100: begin
            // XOR
            alu_enable  = 1;
            rf_write_en = 1;
            alu_op      = 2'b11;
        end

        3'b101: begin
            // AND
            alu_enable  = 1;
            rf_write_en = 1;
            alu_op      = 2'b01;
        end

        3'b110: begin
            // OR
            alu_enable  = 1;
            rf_write_en = 1;
            alu_op      = 2'b10;
        end

        3'b111: begin
            // ADD
            alu_enable  = 1;
            rf_write_en = 1;
            alu_op      = 2'b00;
        end

    endcase

end

endmodule
