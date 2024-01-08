/*
 * Copyright 2021 The CFU-Playground Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "proj_menu.h"

#include <stdio.h>

#include "cfu.h"
#include "menu.h"

namespace {

// Template Fn
void do_hello_world(void) { puts("Hello, World!!!\n"); }

// Test template instruction
void do_exercise_cfu_op0(void) {
  puts("\r\nExercise CFU Op0 aka ADD\r\n");

  unsigned int a = 0;
  unsigned int b = 0;
  unsigned int cfu = 0;
  unsigned int count = 0;
  unsigned int pass_count = 0;
  unsigned int fail_count = 0;

  for (a = 0x00004567; a < 0xF8000000; a += 0x00212345) {
    for (b = 0x0000ba98; b < 0xFF000000; b += 0x00770077) {
      cfu = cfu_op0(0, a, b);
      if (cfu != a + b) {
        printf("[%4d] a: %08x b:%08x a+b=%08x cfu=%08x FAIL\r\n", count, a, b,
               a + b, cfu);
        fail_count++;
      } else {
        pass_count++;
      }
      count++;
    }
  }

  printf("\r\nPerformed %d comparisons, %d pass, %d fail\r\n", count,
         pass_count, fail_count);
}

void test_op(void)
{
  // printf("test op1\n");
  cfu_op1(0,128,0);
  for(int k=0;k<10;k++){
    for(int j=0;j<10;j++){
      cfu_op5(0,0,(j%2)+1);
      for(int i=0;i<128;i++){
        // cfu_op2(0,1000+i,100+i);
        uint8_t a1 = 100+i-k;
        uint8_t a2 = -20+i*5;
        uint8_t a3 = 100-10*i+j;
        uint8_t a4 = 25+8*i-j;
        uint32_t a = (a1<<24) | (a2<<16) | (a3<<8) | a4;

        uint8_t b1 = -100+i+8*j;
        uint8_t b2 = 20+i-j+k;
        uint8_t b3 = -100-10*i-2*k;
        uint8_t b4 = 11+8*i;
        uint32_t b = (b1<<24) | (b2<<16) | (b3<<8) | b4;

        cfu_op2(0,a,b);


      }
      // printf("exit1\n");
      // cfu_op2(0,10,20);
      // printf("exit2\n");
      // cfu_op2(0,400,600);
      // printf("exit3\n");
      // cfu_op2(0,1111,2222);
      cfu_op6(0,0,0);
      // printf("exit4\n");.

      // while(1){
      //     int32_t tmp = cfu_op7(0,0,0);
      //     printf("in while\n");
      //     // printf("in op7 and the return value = %ld\n",tmp);
      //     if(tmp==1){
      //       break;
      //     }
      // }


      // printf("exit5\n");
      // for(int idx = 0; idx<4 ; idx++){
      //   for(int idy = 0; idy < 4; idy++){
      //     int32_t result_tmp = cfu_op3(0,idx,idy);
      //     result_tmp = cfu_op3(0,idx,idy);
      //     // result_tmp = cfu_op3(0,idx,idy);
      //     printf("%ld ",result_tmp);
      //   }
      //   printf("\n");
      // }
      cfu_op4(0,0,0);
      // printf("exit6\n");
    }
  }
    for(int j=0 ; j<10 ; j++){
      for(int idx = 0; idx<4 ; idx++){
        for(int idy = 0; idy < 4; idy++){
          int32_t result_tmp = cfu_op3(0,j*4+idx,idy);
          result_tmp = cfu_op3(0,j*4+idx,idy);
          // result_tmp = cfu_op3(0,idx,idy);
          printf("%ld ",result_tmp);
        }
        printf("\n");
      }
    }
  // cfu_op7(0,0,0);
}

void test_cal(void)
{ 
  // long inputs,weights;
  // inputs = 789456123;
  // weights = 13498749;
  // printf("before op2\n");
  // cfu_op2(0,inputs,weights);
  // printf("after op2\n");
  // printf("after op2\n");
  // printf("after op2\n");
  // printf("after op2\n");
  // printf("after op2\n");
  // printf("after op2\n");

  // // long long tmp = cfu_op3(0,0,1);
  // printf("%ld\n",cfu_op3(0,0,2));

  // printf("op2 and op3\n");
  // printf("%ld\n",cfu_op3(0,0,1));
  // int input[4][5] = {
  //   {1,2,3,4,2},
  //   {5,6,7,8,7},
  //   {9,10,11,12,2},
  //   {13,14,15,16,1}
  // };
  uint8_t input[4][4] = {
    {1,2,3,4},
    {5,6,7,8},
    {9,10,11,12},
    {13,14,15,16}
  };
  // int weight[5][6] = {
  //   {2,5,6,9,5,44},
  //   {3,10,2,55,11,52},
  //   {8,42,26,12,64,42},
  //   {10,25,55,63,89,104},
  //   {11,100,205,888,996,291}
  // };
  int8_t weight[4][4] = {
    {2,5,6,9},
    {3,-10,2,55},
    {10,25,55,63},
    {11,100,-124,-25}
  };

  for(int i=0; i<4 ; i = i+4){
    for(int j=0 ; j<4 ; j = j+4){
      for(int k=0 ; k<4; k = k+ 1 ){
        printf("%d %d %d %d\n",weight[k][j],weight[k][j+1],weight[k][j+2],weight[k][j+3]);
        uint8_t input0 = input[i][k];
        uint8_t input1 = input[i+1][k];
        uint8_t input2 = input[i+2][k];
        uint8_t input3 = input[i+3][k];
        uint32_t test_a = (input0<<24) | (input1<<16) | (input2<<8) | (input3);

        uint8_t weight0 = weight[k][j];
        uint8_t weight1 = weight[k][j+1];
        uint8_t weight2 = weight[k][j+2];
        uint8_t weight3 = weight[k][j+3];
        uint32_t test_b = (weight0<<24) | (weight1<<16) | (weight2<<8)| (weight3) ;
        cfu_op2(0,test_a,test_b);
      }
      // cfu_op4(0,0,0);   
    }
  }
  for(int idx=0;idx<4;idx++){
        for(int idy=0;idy<4;idy++){
          int32_t tmp = cfu_op3(0,idx,idy);
          printf("%ld ",tmp);
        }
        printf("\n");
  }
  
}


struct Menu MENU = {
    "Project Menu",
    "project",
    {
        MENU_ITEM('0', "exercise cfu op0", do_exercise_cfu_op0),
        MENU_ITEM('h', "say Hello", do_hello_world),
        MENU_ITEM('t', "test op", test_op),
        MENU_ITEM('k', "test op cal", test_cal),
        MENU_END,
    },
};
};  // anonymous namespace

extern "C" void do_proj_menu() { menu_run(&MENU); }
