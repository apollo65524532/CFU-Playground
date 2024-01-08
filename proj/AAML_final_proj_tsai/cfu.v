// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "/home/tsaijb/CFU-Playground/proj/my_tpu/TPU.v"

module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,//I add signed
  input      [31:0]   cmd_payload_inputs_1,//I add signed
  output              rsp_valid,
  input               rsp_ready,
  output     [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

  //wire [9:0] cmd_payload_function_id;
  wire [2:0] my_func;
  wire data_in,data_out;
  wire [31:0] tmp_input0,tmp_input1;
  wire  [31:0] C_output;
  // wire back;
  reg rsp_valid_reg;
  reg [31:0] out_reg;


  wire [15:0] tmp_out;
  wire C_out_signal;

  wire [31:0] Cout;

  TPU tpu(
    .clk(clk),
    .rst_n(reset),
    .funct(my_func),
    .input0(tmp_input0),
    .input1(tmp_input1),
    .C_output(C_output),
    .data_in(data_in),
    .data_out(data_out),
    // .back(back),
    .idxout(tmp_out),
    .C_out_signal(C_out_signal),
    .Cout(Cout)
  );

// assign rsp_payload_outputs_0 = (my_func==3'd3) ? C_output : (my_func==3'd7)? tmp_out : 1'b0; //here is my deleted
// assign rsp_payload_outputs_0 = (C_out_signal) ? C_output : (my_func==3'd7)? tmp_out : 1'b0; 
assign rsp_payload_outputs_0 = (my_func==3'd3) ? Cout : (my_func==3'd7)? tmp_out : (my_func==3'd1)? tmp_input1 : 0; 

// assign rsp_payload_outputs_0 = (my_func==3'd2)? tmp_out : 1'b0;
assign cmd_ready = rsp_ready;
assign rsp_valid = (cmd_valid && my_func!=0) ? 1'b1 : 1'b0;



assign my_func = (cmd_ready && cmd_valid) ? cmd_payload_function_id[2:0] : 1'b0;

// assign tmp_input0 = (my_func==3'd2 || my_func==3'd3 || my_func==3'd5) ? (cmd_payload_inputs_0):1'b0;//signed delete
// assign tmp_input1 = (my_func==3'd2 || my_func==3'd3) ? (cmd_payload_inputs_1):1'b0;
assign tmp_input0 = (cmd_payload_inputs_0[31:0]);
assign tmp_input1 = (cmd_payload_inputs_1[31:0]);





endmodule



