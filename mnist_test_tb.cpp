#include <stdio.h>
#include "tt_nn.h"


void mnist_forward(
    TYPE_DATA* data_input_output_buffer,
    TYPE_WEIGHT** weight,
    TYPE_WEIGHT** bias,
    int batchsize
){
    
    int input_shape[] = {7, 4, 7, 4};
    int hidden_shape0[] = {4, 8, 4, 4};
    int rank0[] = {20, 20, 20};
    int hidden_shape1[] = {32, 16};
    
    int rank1[] = {20};
    int output_shape[] = {2, 5};

    for (int i = 0; i < batchsize; i++) {
        tensor_train_forward(
            data_input_output_buffer0,
            weight[0],
            bias[0],
            i * 28*28,
            28*28*batchsize + i * 512,
            28*28**batchsize + 512*batchsize + 10 * batchsize,
            input_shape,
            hidden_shape0,
            rank0,
            4,
			8*8*20*20,
			20*32*32
        );
        relu_inplace(hidden, 512);
        tensor_train_forward(
            hidden,
            weight[1],
            bias[1],
            28*28*batchsize + i * 512,
            28*28*batchsize + 512*batchsize + i * 10,
            28*28*batchsize + 512*batchsize + 10 * batchsize,
            hidden_shape1,
            output_shape,
            rank1,
            2,
			8*8*20*20,
			20*32*32
        );
    }

}

template<typename T>
void load_file(T* data, const char* filename, int size) {
    float* buffer = new float[size];
    FILE* f = fopen(filename, "rb");
    fread((void*)buffer, sizeof(float), size, f);
    fclose(f);
    for(int i = 0; i < size; i++) {
        data[i] = T(buffer[i]);
    }
}

template<typename T>
void save_file(T* data, const char* filename, int size) {
    float* buffer = new float[size];
    for(int i = 0; i < size; i++) {
        buffer[i] = float(data[i]);
    }
    FILE* f = fopen(filename, "rb");
    fwrite((void*)buffer, sizeof(float), size, f);
    fclose(f);
}

int main(){
    TYPE_DATA* data_input_output_buffer = new TYPE_DATA[4800000];
    TYPE_WEIGHT* weight[2];
    TYPE_DATA* bias[2];
    weight[0] = new TYPE_WEIGHT[8*8*20*20*4];
    weight[1] = new TYPE_WEIGHT[8*8*20*20*2];
    bias[0] = new TYPE_DATA[512];
    bias[1] = new TYPE_DATA[10];

    load_file(weight[0], "weight0.bin", 7*4*20);
    load_file(weight[0] + 8*8*20*20, "weight1.bin", 20*4*8*20);
    load_file(weight[0] + 8*8*20*20*2, "weight2.bin", 20*7*4*20);
    load_file(weight[0] + 8*8*20*20*3, "weight3.bin", 20*4*4);
    load_file(weight[1], "weight4.bin", 32*2*20);
    load_file(weight[1] + 8*8*20*20, "weight5.bin", 16*5*20);
    load_file(bias[0], "bias0.bin", 512);
    load_file(bias[1], "bias1.bin", 10)
    load_file(data_input_output_buffer, "input.bin", 28*28*100);

    mnist_forward(data, weight, bias, output);

    save_file(data_input_output_buffer+28*28*100+512*100, 10*100, 100);
    delete[] weight[0];
    delete[] weight[1]
    delete[] bias[0];
    delete[] bias[1];
    deleta[] data_input_output_buffer;
}