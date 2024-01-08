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

// assign prod_extra = (cmd_payload_function_id[2:0] == 3'd0)? ($signed(cmd_payload_function_id[9:3]) + $signed(InputOffset)) * tmp_extra_reg :
//                                                             ($signed(cmd_payload_function_id[9:3]) * (-1) + $signed(InputOffset)) * tmp_extra_reg;
wire signed [31:0] sum_prods;
assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3;// + prod_extra;

// Only not ready for a command when we have a response.
assign cmd_ready = ~rsp_valid;

reg signed [7:0] tmp_extra_reg;



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

// control for tmp_extra_reg
reg tmp_ctrl_cnt;
always @(posedge clk) begin
  if (reset) begin
    tmp_ctrl_cnt <= 0;
  end 
  
  else if (cmd_valid) begin
    if (cmd_payload_function_id[2:0] == 3'd0) begin
      tmp_ctrl_cnt <= ~tmp_ctrl_cnt;
    end

    else if (cmd_payload_function_id[2:0] == 3'd1) begin
      tmp_ctrl_cnt <= ~tmp_ctrl_cnt;
    end

    else begin
      tmp_ctrl_cnt <= tmp_ctrl_cnt;
    end
  end
end


always @(posedge clk) begin
  if (reset) begin
    tmp_extra_reg <= 8'b0;
  end 
  
  else if (cmd_valid) begin
    if (cmd_payload_function_id == {7'd0, 3'd2}) begin
      tmp_extra_reg <= 0;
    end

    else if (cmd_payload_function_id[2:0] == 3'd0) begin
      tmp_extra_reg <= (tmp_ctrl_cnt == 1)? 0 : $signed(cmd_payload_function_id[9:3]);
    end

    else if (cmd_payload_function_id[2:0] == 3'd1) begin
      tmp_extra_reg <= (tmp_ctrl_cnt == 1)? 0 : (-1) * $signed(cmd_payload_function_id[9:3]);
    end

    else begin
      tmp_extra_reg <= tmp_extra_reg;
    end
  end
end

always @(posedge clk) begin
  if (reset) begin
    rsp_payload_outputs_0 <= 32'b0;
  end 
  
  else if (cmd_valid) begin
    case (cmd_payload_function_id[2:0])
      3'd0: rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
      3'd1: rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
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