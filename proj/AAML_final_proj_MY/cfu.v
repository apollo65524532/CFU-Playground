module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

reg [15:0] InputOffset;

// SIMD multiply step:
wire signed [16:0] prod_0, prod_1, prod_2, prod_3, prod_extra;
assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + $signed(InputOffset))
                * $signed(cmd_payload_inputs_1[7 : 0]);
assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + $signed(InputOffset))
                * $signed(cmd_payload_inputs_1[15: 8]);
assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + $signed(InputOffset))
                * $signed(cmd_payload_inputs_1[23:16]);
assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + $signed(InputOffset))
                * $signed(cmd_payload_inputs_1[31:24]);

reg [31:0] tmp_in1, tmp_in2;
wire signed [16:0] extra1 = ($signed(tmp_in1[31:24]) + $signed(InputOffset))
              * $signed(tmp_in2[31:24]);

wire signed [16:0] extra2 = ($signed(tmp_in1[23:16]) + $signed(InputOffset))
              * $signed(tmp_in2[23:16]);

wire signed [16:0] extra3 = ($signed(tmp_in1[15:8]) + $signed(InputOffset))
              * $signed(tmp_in2[15:8]);

wire signed [16:0] extra4 = ($signed(tmp_in1[7:0]) + $signed(InputOffset))
              * $signed(tmp_in2[7:0]);

wire signed [31:0] sum_prods, sum_prods_8group;

assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3; 
assign sum_prods_8group = sum_prods + extra1 + extra2 + extra3 + extra4;

// Only not ready for a command when we have a response.
assign cmd_ready = ~rsp_valid;


always @(posedge clk) begin
  if (reset) begin
    rsp_valid <= 1'b0;
  end 
  
  else if (rsp_valid) begin
    // Waiting to hand off response to CPU.
    rsp_valid <= ~rsp_ready;
  end 
  
  else if (cmd_valid) begin
    rsp_valid <= 1'b1;
  end
end

always @(posedge clk) begin
  if (reset) begin
    tmp_in1 <= 32'b0;
    tmp_in2 <= 32'b0;
  end 
  
  else if (cmd_valid) begin
    case (cmd_payload_function_id[2:0])
      3'd0: begin
        tmp_in1 <= 32'b0;
        tmp_in2 <= 32'b0;
      end

      3'd1: begin
        tmp_in1 <= cmd_payload_inputs_0;
        tmp_in2 <= cmd_payload_inputs_1;
      end

      3'd2: begin
        tmp_in1 <= 32'b0;
        tmp_in2 <= 32'b0;
      end

      default: begin
        tmp_in1 <= 32'b0;
        tmp_in2 <= 32'b0;
      end
    endcase
  end
end

always @(posedge clk) begin
  if (reset) begin
    rsp_payload_outputs_0 <= 32'b0;
  end 
  
  else if (cmd_valid) begin
    case (cmd_payload_function_id[2:0])
      3'd0: rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods_8group;
      3'd1: rsp_payload_outputs_0 <= 0;
      3'd3: rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
      3'd2: rsp_payload_outputs_0 <= 0;
      default: rsp_payload_outputs_0 <= 0;
    endcase
  end
end

always @(posedge clk) begin
  if (reset) begin
    InputOffset <= 0;
  end 
  
  else if (cmd_valid) begin
    case (cmd_payload_function_id)
      {7'd0, 3'd0}: InputOffset <= InputOffset;
      {7'd0, 3'd1}: InputOffset <= InputOffset;
      {7'd0, 3'd2}: InputOffset <= cmd_payload_inputs_0[15:0];
      default: InputOffset <= InputOffset;
    endcase
  end
end



endmodule