module AB_control(
    input clk,
    input rst_n,
    input [31:0] input0,
    input [31:0] input1,
    input [2:0] funct,
    input unsigned [15:0] K_,//unsigned I add
    input [7:0] state,
    output reg A_wr_en,
    output reg B_wr_en,
    output reg [15:0] A_index,
    output reg [15:0] B_index,
    output reg [15:0] A_idx_out,
    output reg [15:0] B_idx_out,
    output reg [31:0] A_data_in,
    output reg [31:0] B_data_in
);


localparam INIT = 2'b00;
localparam DONE = 2'b01;

reg [15:0] A_idx_in,B_idx_in;

reg [15:0] B_idx_out_reg,A_idx_out_reg;



// control signal for buffer
always@(posedge clk)begin
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

always@(posedge clk )begin
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
always@(posedge clk )begin
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
    if(rst_n || funct==3'd4)begin
        A_idx_in <= 0;
    end
    else if(funct==3'd2)begin
        A_idx_in <= A_idx_in + 1'b1;
    end
end

always@(posedge clk)begin
    if(rst_n||funct==3'd4)begin
        // B_idx_in <= 0;
        B_index <= 0;
    end
    else if(funct==3'd2)begin
        B_index <= B_idx_in;
    end
end

always@(posedge clk)begin
    if(rst_n || funct==3'd4)begin
        B_idx_in <= 0;
    end
    if(funct==3'd2)begin
        B_idx_in <= B_idx_in + 1'b1;
    end
end

// wire quan_A1[15:0];
// wire quan_A2[15:0];
// wire quan_A3[15:0];
// wire quan_A4[15:0];
// assign quan_A1 = input0[31:24] + input_offset;
// assign quan_A2 = input0[23:16] + input_offset;
// assign quan_A3 = input0[15:8] + input_offset;
// assign quan_A4 = input0[7:0] + input_offset;

//posedge sends signal,negedge sends data
always@(posedge clk)begin//neg to pos
    if(rst_n)begin
        A_data_in <= 0;
    end
    else if(funct==3'd2)begin
        A_data_in <= input0;
    //     A_data_in[63:48] <= quan_A1;
    //     A_data_in[47:32] <= quan_A2;
    //     A_data_in[31:16] <= quan_A3;
    //     A_data_in[15:0] <= quan_A4;
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
    if(rst_n || funct==3'd4 || A_idx_out==(K_+1))begin
        A_idx_out <= 0;
    end
    else if(state == DONE)begin
        A_idx_out <= A_idx_out_reg;
    end
end

always@(posedge clk)begin
    if(rst_n || funct==3'd4 || A_idx_out==(K_+1))begin
        A_idx_out_reg <= 0;
    end
    else if(state == DONE)begin
        A_idx_out_reg <= A_idx_out_reg + 1'b1;
    end
end

// assign go = (A_idx_out_reg > 0)? 1'b1 : 0;


always@(posedge clk )begin
    if(rst_n || funct==3'd4 || B_idx_out == (K_+1))begin
        B_idx_out <= 0;
    end
    else if(state ==DONE)begin
        B_idx_out <= B_idx_out_reg;
    end
end

always@(posedge clk)begin
    if(rst_n || funct==3'd4 || B_idx_out == (K_+1))begin
        B_idx_out_reg <= 0;
    end
    else if(state == DONE)begin
        B_idx_out_reg <= B_idx_out_reg + 1'b1;
    end
end

// always@(posedge clk )begin
//     if(rst_n)begin
//         A_rows <= 0;
//     end
//     else if(funct==3'd6)begin
//         A_rows <= A_rows + 1;
//     end
// end


endmodule