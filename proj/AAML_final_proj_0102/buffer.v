
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