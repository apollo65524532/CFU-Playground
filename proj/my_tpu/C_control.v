module C_control(
    input clk,
    input rst_n,
    input [2:0] funct,
    input [31:0] input0,
    input [31:0] input1,
    input C_in_signal,
    input [3:0] count,
    output reg [15:0] C_idx_in,
    output reg [15:0] C_idx_out,
    output reg C_wr_en
);


reg [15:0] C_idx;

reg delay;


//OUT2C wr_en sets to 1
always@(posedge clk )begin
    if(rst_n || funct==3'd1)begin
        C_wr_en <= 0;
    end
    // else if(funct==3'd6)begin
    //     C_wr_en <= 0;
    // end
    else if(C_in_signal && count>0 && delay==0 )begin
        C_wr_en <= 1'b1;
    end
    else if(delay)begin
       C_wr_en <= 0; 
    end
    else begin
        C_wr_en <= 0;
    end
end

always@(posedge clk)begin
    if(rst_n || funct==3'd1)begin
        delay <= 0;
    end
    else if(C_in_signal && count==4)begin
        delay <= 1'b1;
    end
    else if(count==0)begin
        delay <= 0;
    end

end

always@(posedge clk )begin
    if(rst_n || funct ==3'd1)begin
        C_idx_in <= 0;
    end
    else if(C_in_signal && count <= 4 && count>0)begin//C_in_signal to C_wr_en
        // C_idx_in <= (C_idx-1) + (A_rows-1)*4;
        C_idx_in <= (C_idx-1);
    end        
end

always@(posedge clk)begin
    if(rst_n || funct==3'd1)begin
        C_idx <= 1'b0;
    end
    else if(C_in_signal && count < 4 )begin//C_in_signal to C_wr_en
        C_idx <= C_idx + 1'b1;
    end
end



always@(posedge clk )begin
    if(rst_n || funct==3'd1)begin
        C_idx_out <= 0;
    end
    //out idx
    else if(funct==3'd3)begin
        // C_idx_out <= input0 + 4*C_out_rows;
        C_idx_out <= input0;
    end
end


endmodule