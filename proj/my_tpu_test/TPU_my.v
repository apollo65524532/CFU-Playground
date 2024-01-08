module TPU(
    clk,
    rst_n,

    funct,
    input0,
    input1,
    C_output,
    data_in,
    data_out,

    // output back to cpu
    idxout, //new
    C_out_signal,
    Cout
);


input clk;
input rst_n;
input data_in;
input data_out;
input [2:0] funct;
input [31:0] input0,input1;
output reg C_out_signal;
output reg    [31:0] C_output;
// output reg back;

output [15:0] idxout;
output reg [31:0] Cout;


//* Implement your design here

parameter IDLE = 2'd0;
parameter READ = 2'd1;
parameter OUTPUT = 2'd2;

reg [15:0] mula_temp [0:15];
reg [15:0] mulb_temp [0:15];
reg [31:0] psum_temp [0:15];
     
// reg [7:0] a1_temp;
// reg [7:0] a2_temp [0:1];
// reg [7:0] a3_temp [0:2];
// reg [7:0] a4_temp [0:3];
     
// reg [7:0] b1_temp;
// reg [7:0] b2_temp [0:1];
// reg [7:0] b3_temp [0:2];
// reg [7:0] b4_temp [0:3];

reg [7:0] a1_temp;
reg [7:0] a2_temp;
reg [7:0] a3_temp;
reg [7:0] a4_temp;
     
reg [7:0] b1_temp;
reg [7:0] b2_temp;
reg [7:0] b3_temp;
reg [7:0] b4_temp;

reg [15:0] mul [0:15];

reg [1:0] state_cs, state_ns;
reg [7:0] row_offset;
reg [7:0] col_offset;
reg [7:0] counter_a;
reg [7:0] counter_b;
reg [31:0] counter;
reg [31:0] counter_out;
reg [15:0] index_a;
reg [15:0] index_b;
reg [15:0] index_c;
reg           A_wr_en_r;
reg [15:0]    A_index_r;
reg [31:0]    A_data_in_r;

reg           B_wr_en_r;
reg [15:0]    B_index_r;
reg [31:0]    B_data_in_r; 

reg           C_wr_en_r;
reg [15:0]    C_index_r;
reg [127:0]   C_data_in_r;

reg [2:0] out_limit;

reg [7:0]      K_tmp;
reg [7:0]      M_tmp;
reg [7:0]      N_tmp;

integer i;

global_buffer_AB #(
    .ADDR_BITS(16),
    .DATA_BITS(32)
)
gbuff_A1(
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out_wire),
    .index_out(A_idx_out),
    .out(out_signal)
);


global_buffer_AB #(
    .ADDR_BITS(16),
    .DATA_BITS(32)
) 
gbuff_B1(
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(B_wr_en),
    .index(B_index),
    .data_in(B_data_in),
    .data_out(B_data_out_wire),
    .index_out(B_idx_out),
    .out(out_signal)
);

global_buffer_C #(
    .ADDR_BITS(16),
    .DATA_BITS(128)
) gbuff_C(
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(C_wr_en),
    .index(C_index),
    .data_in(C_data_in),
    .data_out(C_data_out),
    .index_out(C_idx_out),
    .out(C_out_signal)
);

//* Implement your design here

// -----------------------------Finite State Machine------------------------------//
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) state_cs <= IDLE;
    else state_cs <= state_ns;
  end

  always @(*) begin
    case (state_cs)
      IDLE: state_ns = (in_valid || busy) ? READ : IDLE;
      READ: state_ns = (counter <= (K_tmp + 6)) ? READ : OUTPUT; //including calculation
      OUTPUT: state_ns = (counter_out < 4) ? OUTPUT : IDLE;
      default: state_ns = IDLE;
    endcase
  end

//-----------------------------K_tmp, M_tmp and N_tmp------------------------------//
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
      K_tmp <= 0;
    end

    else if(in_valid)begin
      K_tmp <= K;
    end

    else if(state_ns == IDLE)begin
      K_tmp <= K_tmp;
    end
  end

// -----------------------------offset------------------------------//
  always @(*) begin
    row_offset = 1;
    col_offset = 1;
  end
  
  
// -----------------------------data from A B buffer------------------------------//
  always@(negedge clk)begin
      if(rst_n)begin
          A_data_out <= 0;
      end

      else begin
          A_data_out <= A_data_out_wire;
      end
  end

  always@(negedge clk)begin
      if(rst_n)begin
          B_data_out <= 0;
      end

      else begin
          B_data_out <= B_data_out_wire;
      end
  end


// -----------------------------write enable------------------------------//
  assign A_wr_en = 0;
  assign B_wr_en = 0;
  assign C_wr_en = (state_ns == OUTPUT)? 1:0;
  

  
// -----------------------------counter------------------------------//
//coounter_read
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) counter <= 0;
    else begin
      if(state_cs == READ)
        counter <= counter + 1;
      else
        counter <= 0;
    end
  end

// counter_out
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) counter_out <= 0;
    else begin
      if(state_cs == OUTPUT)
        counter_out <= counter_out + 1;
      else
        counter_out <= 0;
    end
  end

// counter_a
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) counter_a <= 0;
    else begin
      if(counter_out == out_limit-1 && state_cs == OUTPUT) begin
        if(counter_a < col_offset)
          counter_a <= counter_a + 1;
        else
          counter_a <= 1;
      end
      else if(state_ns == IDLE)
        counter_a <= 0;
    end
  end

// counter_b
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) counter_b <= 0;
    else begin
      if(state_ns == IDLE && counter_a == col_offset && busy)
        counter_b <= counter_b + 1;
      else if(state_ns == IDLE)
        counter_b <= 0;
    end
  end

// -----------------------------index------------------------------//
// buffer index
  assign A_index = index_a;
  assign B_index = index_b;
  assign C_index = (!rst_n)? 0:index_c;
// index_a
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) index_a <= 15'd0;
    else begin
      if(state_cs == OUTPUT) begin
        
        if(K_tmp == 1)
          index_a <= 1;
        else if(state_ns==IDLE )begin
          index_a <= index_a + 15'd1;
        end
        else
          index_a <= index_a;
      end
      else if(state_cs == IDLE && counter_a == col_offset)begin
          index_a <= 0;
      end
      else if(state_ns == IDLE)begin
        index_a <= 15'd0;
      end
      else if(state_cs == READ) begin
        if(counter < (K_tmp - 1))
          index_a <= index_a + 15'd1;
      end
      else begin
        index_a <= index_a;
      end
    end
  end

// index_b
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) index_b <= 0;
    else begin
      if(state_cs == IDLE && busy) begin
        index_b <= K_tmp * counter_b;
      end
      else if(state_ns == IDLE)begin
        index_b <= 15'd0;
      end
      else if(state_cs == READ) begin
        if(counter < (K_tmp - 1))
          index_b <= index_b + 1;
      end
    end
  end
  
// index_c
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) index_c <= 0;
    else begin
      if(state_ns == IDLE)
        index_c <= 0;
      else if(state_cs == OUTPUT && state_ns == OUTPUT)
        index_c <= index_c + 1;
      else
        index_c <= index_c;
    end
  end
  
// -----------------------------buffer data in------------------------------//
//only global buffer c has data input
  assign A_data_in = 0;
  assign B_data_in = 0;

  assign C_data_in = (!rst_n)? 0:{psum_temp[(counter_out << 2) ], psum_temp[(counter_out << 2) + 5'd1], psum_temp[(counter_out << 2) + 5'd2], psum_temp[(counter_out << 2)+5'd3]};


// -----------------------------multiplicand and multiplier------------------------------//
//propagate to next row
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      for(i = 0; i < 16; i = i + 1)
        mula_temp[i] <= 8'd0;
    end
    else begin
      if(state_cs == READ) begin
        mula_temp[0] <= a1_temp;
        mula_temp[1] <= mula_temp[0];
        mula_temp[2] <= mula_temp[1];
        mula_temp[3] <= mula_temp[2];

        mula_temp[4] <= a2_temp;
        mula_temp[5] <= mula_temp[4];
        mula_temp[6] <= mula_temp[5];
        mula_temp[7] <= mula_temp[6];

        mula_temp[8] <= a3_temp;
        mula_temp[9] <= mula_temp[8];
        mula_temp[10] <= mula_temp[9];
        mula_temp[11] <= mula_temp[10];

        mula_temp[12] <= a4_temp;
        mula_temp[13] <= mula_temp[12];
        mula_temp[14] <= mula_temp[13];
        mula_temp[15] <= mula_temp[14];
      end
    end
  end
//propagate to next column
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      for(i = 0; i < 16; i = i + 1)
        mulb_temp[i] <= 8'd0;
    end

    else if(state_cs == READ) begin
      mulb_temp[0] <= b1_temp;
      mulb_temp[4] <= mulb_temp[0];
      mulb_temp[8] <= mulb_temp[4];
      mulb_temp[12] <= mulb_temp[8];

      mulb_temp[1] <= b2_temp;
      mulb_temp[5] <= mulb_temp[1];
      mulb_temp[9] <= mulb_temp[5];
      mulb_temp[13] <= mulb_temp[9];

      mulb_temp[2] <= b3_temp;
      mulb_temp[6] <= mulb_temp[2];
      mulb_temp[10] <= mulb_temp[6];
      mulb_temp[14] <= mulb_temp[10];

      mulb_temp[3] <= b4_temp;
      mulb_temp[7] <= mulb_temp[3];
      mulb_temp[11] <= mulb_temp[7];
      mulb_temp[15] <= mulb_temp[11];
    end
  end
  
  
// -----------------------------Multiplication------------------------------//
  always @(*) begin
    for(i = 0; i < 16; i = i + 1)
      mul[i] = mula_temp[i] * mulb_temp[i];
  end

// PE SUM update
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
      for(i = 0; i < 16; i = i + 1)
        psum_temp[i] <= 8'd0;
    end
    else begin
      if(state_cs == READ) begin
        for(i = 0; i < 16; i = i + 1)
          psum_temp[i] <= psum_temp[i] + mul[i];
      end
      else if(state_cs == IDLE) begin
        for(i = 0; i < 16; i = i + 1)
          psum_temp[i] <= 8'd0;
      end

      else psum_temp[0] = psum_temp[0];

    end
  end


// -----------------------------Data loader A------------------------------//
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) a1_temp <= 0;
    else begin
      if(state_cs == READ) begin
        if(counter < K_tmp)
          a1_temp <= A_data_out[31:24];
        else
          a1_temp <= 0;
      end

      else a1_temp <= a1_temp;
      
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) a2_temp <= 0;
    else begin
      if(state_cs == READ) begin
        if(counter < K_tmp)
          a2_temp <= A_data_out[23:16];
        else
          a2_temp <= 0;
      end

      else a2_temp <= a2_temp;
      
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) a3_temp <= 0;
    else begin
      if(state_cs == READ) begin
        if(counter < K_tmp)
          a3_temp <= A_data_out[15:8];
        else
          a3_temp <= 0;
      end

      else a3_temp <= a3_temp;
      
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) a4_temp <= 0;
    else begin
      if(state_cs == READ) begin
        if(counter < K_tmp)
          a4_temp <= A_data_out[7:0];
        else
          a4_temp <= 0;
      end

      else a4_temp <= a4_temp;
      
    end
  end
  
// -----------------------------Data loader B------------------------------//
  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) b1_temp <= 0;
    else begin
      if(state_cs == READ)
        if(counter < K_tmp)
          b1_temp <= B_data_out[31:24];
        else
          b1_temp <= 0;
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) b2_temp <= 0;
    else begin
      if(state_cs == READ)
        if(counter < K_tmp)
          b2_temp <= B_data_out[23:16];
        else
          b2_temp <= 0;
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) b3_temp <= 0;
    else begin
      if(state_cs == READ)
        if(counter < K_tmp)
          b3_temp <= B_data_out[15:8];
        else
          b3_temp <= 0;
    end
  end

  always @(posedge clk or negedge rst_n) begin
    if(!rst_n) b4_temp <= 0;
    else begin
      if(state_cs == READ)
        if(counter < K_tmp)
          b4_temp <= B_data_out[7:0];
        else
          b4_temp <= 0;
    end
  end





endmodule


module global_buffer_AB #(parameter ADDR_BITS=16, parameter DATA_BITS=32,parameter idx=0)(clk, rst_n, wr_en, index, data_in, data_out,index_out,out);
  input clk;
  input rst_n;
  input wr_en; // Write enable: 1->write 0->read
  input      [ADDR_BITS-1:0] index,index_out;
  input      [DATA_BITS-1:0]       data_in;
  input out;
  output     reg [DATA_BITS-1:0]       data_out;

  // integer i;

  //parameter DEPTH = 2**ADDR_BITS;
  parameter DEPTH = 2048;//65536
//----------------------------------------------------------------------------//
// Global buffer (Don't change the name)                                      //
//----------------------------------------------------------------------------//
  // reg [`GBUFF_ADDR_SIZE-1:0] gbuff [`WORD_SIZE-1:0];
  (*ram_style = "block"*)reg  [DATA_BITS-1:0] gbuff [DEPTH-1:0];
  // reg [ADDR_BITS-1:0] read_addr_reg; 
//----------------------------------------------------------------------------//
// Global buffer read write behavior                                          //
//----------------------------------------------------------------------------//
  always @ (negedge clk) begin
    // read_addr_reg <= index_out;
    if(wr_en /*&& buf_idx==idx*/) begin
      gbuff[index] <= data_in;
    end
  end

  always@(negedge clk)begin
    if(out /*&& buf_idx_out==idx*/)begin
      data_out <= gbuff[index_out];
    end
  end
endmodule

module global_buffer_C #(parameter ADDR_BITS=16, parameter DATA_BITS=128)(clk, rst_n, wr_en, index, data_in, data_out,index_out,out);

  input clk;
  input rst_n;
  input wr_en; // Write enable: 1->write 0->read
  input      [ADDR_BITS-1:0] index,index_out;
  input      [DATA_BITS-1:0]       data_in;
  input out;
  output     reg [DATA_BITS-1:0]       data_out;

  integer i;

  // parameter DEPTH = 2**ADDR_BITS;
  parameter DEPTH = 16384;//

//----------------------------------------------------------------------------//
// Global buffer (Don't change the name)                                      //
//----------------------------------------------------------------------------//
  // reg [`GBUFF_ADDR_SIZE-1:0] gbuff [`WORD_SIZE-1:0];
  (*ram_style = "block"*)reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];
  // reg [ADDR_BITS-1:0] read_addr_reg; 

//----------------------------------------------------------------------------//
// Global buffer read write behavior                                          //
//----------------------------------------------------------------------------//
  always @ (negedge clk) begin
    // read_addr_reg <= index_out;
    if(wr_en) begin
      gbuff[index] <= data_in;
    end
  end

  always@(negedge clk)begin
    if(out)begin

      data_out <= gbuff[index_out];
    end
  end

  // assign data_out = gbuff[read_addr_reg];

endmodule