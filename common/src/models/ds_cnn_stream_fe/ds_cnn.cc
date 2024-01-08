#include "models/ds_cnn_stream_fe/ds_cnn.h"
#include <stdio.h>
#include "menu.h"
#include "models/ds_cnn_stream_fe/ds_cnn_stream_fe.h"
#include "tflite.h"
#include "models/label/label0_board.h"
#include "models/label/label1_board.h"
#include "models/label/label6_board.h"
#include "models/label/label8_board.h"
#include "models/label/label11_board.h"
#include "models/my_cycles.h"

// Initialize everything once
// deallocate tensors when done
static void ds_cnn_stream_fe_init(void) {
  tflite_load_model(ds_cnn_stream_fe, ds_cnn_stream_fe_len);
  reset_my_cycles();
}

static float* non_stream_classify() {
    printf("Running ds_cnn_stream_fe\n");
    tflite_classify();

    // Process the inference results.
    float* output = tflite_get_output_float();
    return output;
}

static void do_classify() {
    tflite_set_input_zeros_float();
    float* result = non_stream_classify();
    
    long long unsigned mac_cycles = 0;
    mac_cycles = get_my_cycles(); 	
    printf("MAC: %lld\n", mac_cycles);
    
    for(size_t i=0; i<12; i++){
        printf("%d : 0x%lx,\n", i, *(uint32_t*)&result[i]);
        //mac_cycles += get_my_cycles();
    }    
    
}

static void do_classify_label0() {
    puts("Classify label0");
    tflite_set_input(label0_data);
    float* result = non_stream_classify();
    
    long long unsigned mac_cycles = 0;
    mac_cycles = get_my_cycles(); 	
    printf("MAC: %lld\n", mac_cycles);
    
    for(size_t i=0; i<12; i++)
        printf("%d : 0x%lx,\n", i, *(uint32_t*)&result[i]);
}

static void do_classify_label1() {
    puts("Classify label1");
    tflite_set_input(label1_data);
    float* result = non_stream_classify();
    
    long long unsigned mac_cycles = 0;
    mac_cycles = get_my_cycles(); 	
    printf("MAC: %lld\n", mac_cycles);
    
    for(size_t i=0; i<12; i++)
        printf("%d : 0x%lx,\n", i, *(uint32_t*)&result[i]);
}

static void do_classify_label6() {
    puts("Classify label6");
    tflite_set_input(label6_data);
    float* result = non_stream_classify();
    
    long long unsigned mac_cycles = 0;
    mac_cycles = get_my_cycles(); 	
    printf("MAC: %lld\n", mac_cycles);
    
    for(size_t i=0; i<12; i++)
        printf("%d : 0x%lx,\n", i, *(uint32_t*)&result[i]);
}

static void do_classify_label8() {
    puts("Classify label8");
    tflite_set_input(label8_data);
    float* result = non_stream_classify();
    
    long long unsigned mac_cycles = 0;
    mac_cycles = get_my_cycles(); 	
    printf("MAC: %lld\n", mac_cycles);
    
    for(size_t i=0; i<12; i++)
        printf("%d : 0x%lx,\n", i, *(uint32_t*)&result[i]);
}

static void do_classify_label11() {
    puts("Classify label11");
    tflite_set_input(label11_data);
    float* result = non_stream_classify();
    
    long long unsigned mac_cycles = 0;
    mac_cycles = get_my_cycles(); 	
    printf("MAC: %lld\n", mac_cycles);
    
    for(size_t i=0; i<12; i++)
        printf("%d : 0x%lx,\n", i, *(uint32_t*)&result[i]);
}
static struct Menu MENU = {
    "Tests for ds_cnn_stream_fe",
    "ds_cnn_stream_fe",
    {
        MENU_ITEM('1', "Run with zeros input", do_classify),
        MENU_ITEM('2', "Run with label0", do_classify_label0),
        MENU_ITEM('3', "Run with label1", do_classify_label1),
        MENU_ITEM('4', "Run with label6", do_classify_label6),
        MENU_ITEM('5', "Run with label8", do_classify_label8),
        MENU_ITEM('6', "Run with label11", do_classify_label11),
        MENU_END,
    },
};

// For integration into menu system
void ds_cnn_stream_fe_menu() {
  ds_cnn_stream_fe_init();
  menu_run(&MENU);
}
