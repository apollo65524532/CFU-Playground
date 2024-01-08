`include "./C_control.v"
//`include "./AB_control.v"
`include "./buffer.v"

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
output reg [31:0] C_output;
// output reg back;

output [15:0] idxout;
output reg [31:0] Cout;


reg [5:0] i,j;
reg  unsigned [15:0] K_tmp;
// reg unsigned [3:0] K_tmpidx;
reg  unsigned [1:0] K_tmpidx_before;


//store array
wire signed [31:0] input_B[3:0]; 
wire signed [31:0] input_A[3:0];

//A,B,C buffer index
reg [15:0] A_idx_out,B_idx_out,C_idx_out;
// wire [15:0] A_rows;


//CFU data out
reg  signed [31:0] out[0:15];//signed  


//buffer signal control
reg A_wr_en, B_wr_en,C_wr_en;

//data for buffer
reg signed [31:0] A_data_in,B_data_in;
//reg [31:0] A_data_in_ff,B_data_in_ff;

reg [127:0] C_data_in;

//index for buffer
reg [15:0] A_index,B_index,C_index;

//take from buffer
wire [127:0] C_data_out;
// reg [31:0] A_data_out;
// reg signed [31:0] B_data_out;
reg [127:0] tmp_Cbuf[3:0];


//state for A,B buffer
reg [2:0] state_ns, state_cs;
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
// wire [31:0] A_data_out_wire;
// wire signed [31:0] B_data_out_wire;
wire signed [31:0] A_data_out;
wire signed [31:0] B_data_out;

reg signed [15:0] mula_temp [0:15];
reg signed [15:0] mulb_temp [0:15];
reg signed [31:0] psum_temp [0:15];

reg signed [7:0] a1_temp;
reg signed [7:0] a2_temp;
reg signed [7:0] a3_temp;
reg signed [7:0] a4_temp;
     
reg signed [7:0] b1_temp;
reg signed [7:0] b2_temp;
reg signed [7:0] b3_temp;
reg signed [7:0] b4_temp;

// reg [15:0] mul [0:15];
reg signed [31:0] mul[0:15];
reg signed [15:0] img_offset;

global_buffer_AB #(
    .ADDR_BITS(16),
    .DATA_BITS(32)
)
gbuff_A(
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(A_wr_en),
    .index(A_index),
    .data_in(A_data_in),
    .data_out(A_data_out),
    .index_out(A_idx_out),
    .out(out_signal)
);


global_buffer_AB #(
    .ADDR_BITS(16),
    .DATA_BITS(32)
) 
gbuff_B(
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(B_wr_en),
    .index(B_index),
    .data_in(B_data_in),
    .data_out(B_data_out),
    .index_out(B_idx_out),
    .out(out_signal)
);

global_buffer_C #(
    .ADDR_BITS(16),
    .DATA_BITS(128)
) 
gbuff_C(
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(C_wr_en),
    .index(C_index),
    .data_in(C_data_in),
    .data_out(C_data_out),
    .index_out(C_idx_out),
    .out(C_out_signal)
);

//---------------------------------------------------FSM--------------------------------------------------//
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
always @(posedge clk) begin
    if(rst_n)begin
        state_cs <= INIT;
    end

    else begin
        state_cs <= state_ns;
    end
end

always @(*)begin
    // if(rst_n || funct==3'd4)begin
    if(rst_n)begin
        state_ns <= INIT;
    end
    //means buffer has data
    else if(funct==3'd6)begin
        state_ns <= DONE;
    end
    //finish counting
    else if(A_idx_out==(K_tmp+1))begin//K_tmp to K_tmp+1
        state_ns <= INIT;
    end
end

//----------------------------------------------------Data loader AB control logic-------------------------------------------------------------//
reg [15:0] A_idx_in,B_idx_in;
reg [15:0] B_idx_out_reg,A_idx_out_reg;

// control signal for buffer
always @(posedge clk)begin
    if(rst_n)begin
        A_wr_en <= 0;
    end
    // cfu send data,wr_en set to 1
    else if(funct==3'd2)begin
        A_wr_en <= 1'b1;
    end
    else begin
        A_wr_en <= 0;
    end
end

always @(posedge clk)begin
    if(rst_n)begin
        B_wr_en <= 0;
    end
    // cfu send data,wr_en set to 1
    else if(funct==3'd2)begin
        B_wr_en <= 1'b1;
    end
    else begin
        B_wr_en <= 0;
    end
end


//control store index for buffer
always @(posedge clk) begin
    if(rst_n || funct==3'd4)begin
        // A_idx_in <= 0;
        A_index <= 0;
    end

    else if(funct==3'd2)begin
        A_index <= A_idx_in;
    end
end

// negedge idx++
always@(posedge clk)begin
    if(rst_n)begin
        A_idx_in <= 0;
    end

    else if(funct==3'd4)begin
        A_idx_in <= 0;
    end

    else if(funct==3'd2)begin
        A_idx_in <= A_idx_in + 1'b1;
    end
end

always@(posedge clk)begin
    if (rst_n)begin
        B_index <= 0;
    end

    else if (funct==3'd4)begin
        B_index <= 0;
    end

    else if(funct==3'd2)begin
        B_index <= B_idx_in;
    end
end

always@(posedge clk)begin
    if (rst_n)begin
        B_idx_in <= 0;
    end

    else if(funct==3'd4)begin
        B_idx_in <= 0;
    end

    else if(funct==3'd2)begin
        B_idx_in <= B_idx_in + 1'b1;
    end
end

//posedge sends signal,negedge sends data
always@(posedge clk)begin//neg to pos
    if(rst_n)begin
        A_data_in <= 0;
    end
    
    else if(funct==3'd2)begin
        A_data_in <= input0;
    end
end

always@(posedge clk)begin//neg to pos
    if(rst_n)begin
        B_data_in <= 0;
    end

    else if(funct==3'd2)begin
        // B_data_in <= ($signed(input1[31:24])) | ($signed(input1[23:16])) | ($signed(input1[15:8])) | ($signed(input1[7:0]));
        B_data_in <= $signed(input1);
    end
end

// control output index for buffer
always@(posedge clk )begin
    if (rst_n)begin
        A_idx_out <= 0;
    end

    else if (funct==3'd4 || A_idx_out==(K_tmp+1))begin
        A_idx_out <= 0;
    end

    else if(state_cs == DONE)begin
        A_idx_out <= A_idx_out_reg;
    end
end

always@(posedge clk)begin
    if (rst_n)begin
        A_idx_out_reg <= 0;
    end

    else if (funct==3'd4 || A_idx_out==(K_tmp+1))begin
        A_idx_out_reg <= 0;
    end

    else if(state_cs == DONE)begin
        A_idx_out_reg <= A_idx_out_reg + 1'b1;
    end
end

always@(posedge clk )begin
    if (rst_n)begin
        B_idx_out <= 0;
    end

    else if (funct==3'd4 || B_idx_out == (K_tmp+1))begin
        B_idx_out <= 0;
    end

    else if(state_cs ==DONE)begin
        B_idx_out <= B_idx_out_reg;
    end
end

always@(posedge clk)begin
    if (rst_n)begin
        B_idx_out_reg <= 0;
    end

    else if (funct==3'd4 || B_idx_out == (K_tmp+1))begin
        B_idx_out_reg <= 0;
    end

    else if(state_cs == DONE)begin
        B_idx_out_reg <= B_idx_out_reg + 1'b1;
    end
end
//------------------------------------------------------------------------------------------------------------------------------------------//

//-----------------------------------------------------C Buffer control logic------------------------------------------------------------------//
reg [15:0] C_idx;
reg delay;

//OUT2C wr_en sets to 1
always@(posedge clk )begin
    if (rst_n)begin
        C_wr_en <= 0;
    end
    
    else if (funct==3'd1)begin
        C_wr_en <= 0;
    end

    else if (C_in_signal && count>0 && delay==0 )begin
        C_wr_en <= 1'b1;
    end

    else if (delay)begin
        C_wr_en <= 0; 
    end

    else begin
        C_wr_en <= 0;
    end
end

always@(posedge clk)begin
    if (rst_n)begin
        delay <= 0;
    end

    else if (funct==3'd1)begin
        delay <= 0;
    end

    else if (C_in_signal && count==4)begin
        delay <= 1'b1;
    end

    else if (count==0)begin
        delay <= 0;
    end
end

always@(posedge clk )begin
    if (rst_n)begin
        C_index <= 0;
    end

    else if (funct ==3'd1)begin
        C_index <= 0;
    end

    else if (C_in_signal && count <= 4 && count>0)begin//C_in_signal to C_wr_en
        // C_index <= (C_idx-1) + (A_rows-1)*4;
        C_index <= (C_idx-1);
    end        
end

always@(posedge clk)begin
    if (rst_n)begin
        C_idx <= 1'b0;
    end

    else if (funct==3'd1)begin
        C_idx <= 1'b0;
    end

    else if (C_in_signal && count < 4 )begin//C_in_signal to C_wr_en
        C_idx <= C_idx + 1'b1;
    end
end



always@(posedge clk )begin
    if (rst_n)begin
        C_idx_out <= 0;
    end

    else if (funct==3'd1)begin
        C_idx_out <= 0;
    end

    //out idx
    else if (funct==3'd3)begin
        // C_idx_out <= input0 + 4*C_out_rows;
        C_idx_out <= input0;
    end
end
//---------------------------------------------------------------------------------------------------------------------------------------------//
//propagate to next row
  always @(posedge clk) begin
    if(rst_n) begin
      for(i = 0; i < 16; i = i + 1)
        mula_temp[i] <= 8'd0;
    end

    else begin
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

//propagate to next column
  always @(posedge clk) begin
    if(rst_n) begin
      for(i = 0; i < 16; i = i + 1)
        mulb_temp[i] <= 8'd0;
    end

    else begin
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

always @(posedge clk)begin
    if (rst_n)begin
        img_offset <= 0;
    end

    else if (funct==3'd1)begin
        img_offset <= input1;
    end
end

// -----------------------------Multiplication------------------------------//
always @(*) begin
    for(i=0; i<16; i=i+1)
        mul[i] = (mula_temp[i] + img_offset) * mulb_temp[i];
end



always@(posedge clk)begin//pos or neg
    if (rst_n)begin//funct==3'd4 kill
        out_signal <= 0;
    end

    else if (A_idx_out==(K_tmp))begin//funct==3'd4 kill
        out_signal <= 0;
    end

    else if(funct==3'd6)begin
        out_signal <= 1'b1;
    end
end

// get K_tmp from cpu in cfu_op1
always@(posedge clk)begin
    if(rst_n)begin 
        K_tmp <= 1'b0;
    end

    else if(funct==3'd1)begin
        K_tmp <= input0;
    end
end

always@(posedge clk)begin//neg to pos
    if(rst_n)begin
        C_data_in <= 0;
    end

    else if (C_in_signal && count <= 4 && count >0) begin//count>=1 I add and =4 to <=4 
        // C_data_in <= (out[count-1-4*(A_rows-1)][0]<<96) | (out[count-1-4*(A_rows-1)][1]<<64) | (out[count-1-4*(A_rows-1)][2]<<32) | (out[count-1-4*(A_rows-1)][3]);
        C_data_in[127:96] <= out[(count-1)*4];
        C_data_in[95:64]  <= out[(count-1)*4+1];
        C_data_in[63:32]  <= out[(count-1)*4+2];
        C_data_in[31:0]   <= out[(count-1)*4+3];
    end
end

always@(negedge clk)begin
    if (C_out_signal)begin
        // Cout <= $signed(C_data_out[127-32*input1-:32]);
        Cout <= $signed(C_data_out[127-32*input1-:32]);
    end
end



//start to calculate 
// PE SUM update
  always @(negedge clk) begin
    if (rst_n)begin//funct 3'd4 -> 3'd6
        for(i=0;i<16;i=i+1)
            out[i] <= 1'b0;
    end

    else if (funct==3'd4 || funct==3'd1)begin//funct 3'd4 -> 3'd6
        for(i=0;i<16;i=i+1)
            out[i] <= 1'b0;
    end

    //there is data in buf and still calculating
    else if(state_cs==DONE  && A_idx_out <= (K_tmp+1) && A_idx_out >=2 )begin//K_tmp+1>=A_idx>=2
        out[0][0] <= out[0][0] + mul[0];
        out[0][1] <= out[0][1] + mul[1];
        out[0][2] <= out[0][2] + mul[2];
        out[0][3] <= out[0][3] + mul[3];

        out[1][0] <= out[1][0] + mul[4];
        out[1][1] <= out[1][1] + mul[5];
        out[1][2] <= out[1][2] + mul[6];
        out[1][3] <= out[1][3] + mul[7];

        out[2][0] <= out[2][0] + mul[8];
        out[2][1] <= out[2][1] + mul[9];
        out[2][2] <= out[2][2] + mul[10];
        out[2][3] <= out[2][3] + mul[11];

        out[3][0] <= out[3][0] + mul[12];
        out[3][1] <= out[3][1] + mul[13];
        out[3][2] <= out[3][2] + mul[14];
        out[3][3] <= out[3][3] + mul[15];
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

    else if(A_idx_out==(K_tmp+1) && K_tmp!=0 )begin//K_tmp to K_tmp + 1 && count>0
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

