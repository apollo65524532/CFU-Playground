`include "./C_control.v"
`include "./AB_control.v"

module TPU(
    clk,
    rst_n,

    funct,
    input0,
    input1,
    C_output,
    data_in,
    data_out,
    // back,
    idxout,//I add
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


reg [5:0] i,j;
reg  unsigned [15:0] K_;
// reg unsigned [3:0] K_idx;
reg  unsigned [1:0] K_idx_before;


//store array
wire signed [31:0] input_B[3:0]; 
wire        [7:0] input_A[3:0];

//A,B,C buffer index
wire [15:0] A_idx_out,B_idx_out,C_idx_out;
// wire [15:0] A_rows;


//CFU data out
reg  signed [31:0] out[0:15];//signed I add


//buffer signal control
wire A_wr_en, B_wr_en,C_wr_en;

//data for buffer
wire [31:0] A_data_in,B_data_in;
reg [127:0] C_data_in;

//index for buffer
wire [15:0] A_index,B_index,C_index;

//take from buffer
wire [127:0] C_data_out;
reg [31:0] A_data_out;
reg signed [31:0] B_data_out;
reg [127:0] tmp_Cbuf[3:0];


//state for A,B buffer
reg [2:0] state;
localparam INIT = 2'b00;
localparam DONE = 2'b01;

// wire [2:0] C_state;
reg out_signal;
// wire go;
reg C_in_signal;

// reg  [2:0] buf_idx,buf_idx_out;



reg [3:0] count;
// assign idxout = (C_index==4*A_rows)? 1'b1 : 0;
assign idxout = C_index;

//data for buffer2
wire [31:0] A_data_out_wire;
wire signed [31:0] B_data_out_wire;




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


C_control cctr(
    .clk(clk),
    .rst_n(rst_n),
    .input0(input0),
    .input1(input1),
    .funct(funct),
    .C_idx_in(C_index),
    .C_idx_out(C_idx_out),
    .C_wr_en(C_wr_en),
    .C_in_signal(C_in_signal),
    .count(count)
);

AB_control abctr(
    .clk(clk),
    .rst_n(rst_n),
    .K_(K_),
    .state(state),
    .input0(input0),
    .input1(input1),
    .funct(funct),
    .A_wr_en(A_wr_en),
    .B_wr_en(B_wr_en),
    .A_index(A_index),
    .B_index(B_index),
    .A_idx_out(A_idx_out),
    .B_idx_out(B_idx_out),
    .A_data_in(A_data_in),
    .B_data_in(B_data_in)
);



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

//count store times
always@(posedge clk)begin
    if(rst_n)begin
        count <= 0;
    end

    else if(funct==3'd4)begin
        count <= 0;
    end

    else if(C_in_signal && count<4)begin
        count <= count + 1'b1;
    end
end

//switch state 
always@(posedge clk)begin
    // if(rst_n || funct==3'd4)begin
    if(rst_n)begin
        state <= INIT;
    end
    //means buffer has data
    else if(funct==3'd6)begin
        state <= DONE;
    end
    //finish counting
    else if(A_idx_out==(K_+1))begin//K_ to K_+1
        state <= INIT;
    end
end

always@(posedge clk)begin//pos or neg
    if (rst_n)begin//funct==3'd4 kill
        out_signal <= 0;
    end

    else if (A_idx_out==(K_))begin//funct==3'd4 kill
        out_signal <= 0;
    end

    else if(funct==3'd6)begin
        out_signal <= 1'b1;
    end
end

// get K_ from cpu in cfu_op5
always@(posedge clk)begin
    if(rst_n)begin//3'd4 to 3'd1 || funct == 3'd1
        // for(i=0;i<3;i=i+1)
        K_ <= 1'b0;
    end
    else if(funct==3'd1)begin
        K_ <= input0;
    end
end

always@(posedge clk)begin//neg to pos
    if(rst_n)begin
        C_data_in <= 0;
    end
    else if(C_in_signal && count <= 4 && count >0)begin//count>=1 I add and =4 to <=4 
        // C_data_in <= (out[count-1-4*(A_rows-1)][0]<<96) | (out[count-1-4*(A_rows-1)][1]<<64) | (out[count-1-4*(A_rows-1)][2]<<32) | (out[count-1-4*(A_rows-1)][3]);
        C_data_in[127:96] <= out[(count-1)*4];
        C_data_in[95:64] <= out[(count-1)*4+1];
        C_data_in[63:32] <= out[(count-1)*4+2];
        C_data_in[31:0] <= out[(count-1)*4+3];
    end
end


assign input_A[0] = A_data_out[31:24];
assign input_A[1] = A_data_out[23:16];
assign input_A[2] = A_data_out[15:8];
assign input_A[3] = A_data_out[7:0]; 
assign input_B[0] = B_data_out[31:24]; 
assign input_B[1] = B_data_out[23:16];
assign input_B[2] = B_data_out[15:8];
assign input_B[3] = B_data_out[7:0];


always@(negedge clk)begin
    if (C_out_signal)begin
        // Cout <= $signed(C_data_out[127-32*input1-:32]);
        Cout <= $signed(C_data_out[127-32*input1-:32]);
    end
end



//start to calculate 
always@(negedge clk )begin//neg to pos
    if (rst_n)begin//funct 3'd4 -> 3'd6
        for(i=0;i<16;i=i+1)
            out[i] <= 1'b0;
    end

    else if (funct==3'd4 || funct==3'd1)begin//funct 3'd4 -> 3'd6
        for(i=0;i<16;i=i+1)
            out[i] <= 1'b0;
    end

    //there is data in buf and still calculating
    else if(state==DONE  && A_idx_out <= (K_+1) && A_idx_out >=2 )begin//K_+1>=A_idx>=2
        for(i=0 ; i<4 ; i = i+1)
          for(j=0 ; j<4 ; j = j+1)
            out[4*i+j] <= out[i][j] + input_A[i] * input_B[j];
            // out[i][j] <= out[i][j] + input_B[i] * input_A[j];
    end
end


//C_in_signal
always@(posedge clk)begin
    if (rst_n)begin
        C_in_signal <= 0;
    end

    else if (funct==3'd4 || funct==3'd1)begin
        C_in_signal <= 0;
    end

    else if(A_idx_out==(K_+1) && K_!=0 )begin//K_ to K_ + 1 && count>0
        C_in_signal <= 1'b1;
    end
end

//C_out_signal
always@(posedge clk)begin
    if (rst_n) begin
        C_out_signal <= 0;
    end

    else if (funct==3'd4 || funct==3'd1)begin
        C_out_signal <= 0;
    end

    else if(funct==3'd3)begin
        C_out_signal <= 1'b1;
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