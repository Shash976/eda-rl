`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// 4-bit ALU for RTL-to-GDSII / Vivado verification
// Operations selected by sel[1:0]:
//   2'b00 : NAND
//   2'b01 : OR
//   2'b10 : XOR
//   2'b11 : ADD
//////////////////////////////////////////////////////////////////////////////////

module alu_4bit (
    input  wire [3:0] A,
    input  wire [3:0] B,
    input  wire [1:0] sel,
    output reg  [3:0] OUT,
    output reg        carry_out
);

    wire [4:0] add_result;

    assign add_result = {1'b0, A} + {1'b0, B};

    always @(*) begin
        OUT       = 4'b0000;
        carry_out = 1'b0;

        case (sel)
            2'b00: begin
                OUT       = ~(A & B);
                carry_out = 1'b0;
            end

            2'b01: begin
                OUT       = A | B;
                carry_out = 1'b0;
            end

            2'b10: begin
                OUT       = A ^ B;
                carry_out = 1'b0;
            end

            2'b11: begin
                OUT       = add_result[3:0];
                carry_out = add_result[4];
            end

            default: begin
                OUT       = 4'b0000;
                carry_out = 1'b0;
            end
        endcase
    end

endmodule

