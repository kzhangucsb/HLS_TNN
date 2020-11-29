#include <stdio.h>
#include <assert.h>
#include "tt_nn.h"

/*********************************
 * data_input_output_buffer:
 * data_in | hidden | output | tmp (reused by sample)
 * forward: reused by layer
 * backward: output_grad | hidden_grad | tmp(reused by layer)
***********************************/


void mnist_forward(
    TYPE_DATA* data_input_output_buffer,
    TYPE_WEIGHT** weight,
    TYPE_DATA** bias,
    int batchsize
){
    
    int input_shape[] = {7, 4, 7, 4};
    int hidden_shape0[] = {4, 8, 4, 4};
    int rank0[] = {16, 16, 16};
    int hidden_shape1[] = {32, 16};
    
    int rank1[] = {16};
    int output_shape[] = {4, 4};

    for (int i = 0; i < batchsize; i++) {
        tensor_train_forward(
            data_input_output_buffer,
            weight[0],
            bias[0],
            i * 28*28,
            28*28*batchsize + i * 512,
            28*28*batchsize + 512*batchsize + 16*batchsize,
            input_shape,
            hidden_shape0,
            rank0,
            4,
			8*8*20*20,
			20*32*32
        );
        relu_inplace(data_input_output_buffer + 28*28*batchsize + i * 512, 512);
        tensor_train_forward(
            data_input_output_buffer,
            weight[1],
            bias[1],
            28*28*batchsize + i * 512,
            28*28*batchsize + 512*batchsize + i * 16,
            28*28*batchsize + 512*batchsize + 16 * batchsize,
            hidden_shape1,
            output_shape,
            rank1,
            2,
			8*8*20*20,
			20*32*32
        );
    }
}

void mnist_train(
    TYPE_DATA* data_input_output_buffer,
    TYPE_WEIGHT** weight,
    TYPE_DATA** bias,
    TYPE_DATA** weight_grad,
    TYPE_DATA** bias_grad,
    unsigned char* label,
    int batchsize
){
    
    int input_shape[] = {7, 4, 7, 4};
    int hidden_shape0[] = {4, 8, 4, 4};
    int rank0[] = {16, 16, 16};
    int hidden_shape1[] = {32, 16};
    
    int rank1[] = {16};
    int output_shape[] = {4, 4};

    for (int i = 0; i < 1; i++) {
        tensor_train_forward(
            data_input_output_buffer,
            weight[0],
            bias[0],
            i * 28*28,
            28*28*batchsize + i * 512,
            28*28*batchsize + 512*batchsize + 16 * batchsize,
            input_shape,
            hidden_shape0,
            rank0,
            4,
			8*8*20*20,
			20*32*32
        );
        relu_inplace(data_input_output_buffer + 28*28*batchsize + i * 512, 512);
        tensor_train_forward(
            data_input_output_buffer,
            weight[1],
            bias[1],
            28*28*batchsize + i * 512,
            28*28*batchsize + 512*batchsize + i * 16,
            28*28*batchsize + 512*batchsize + 16 * batchsize,
            hidden_shape1,
            output_shape,
            rank1,
            2,
			8*8*20*20,
			20*32*32
        );
        // backward
        softmax_ce_grad(
            data_input_output_buffer, 
            label[i], 
            28*28*batchsize + 512*batchsize + i * 16,
            28*28*batchsize + 512*batchsize + 16 * batchsize,
            16
        );
        tensor_train_backward(
            data_input_output_buffer,
            weight[1],
            weight_grad[1],
            28*28*batchsize + 512 * i,
            28*28*batchsize + 512*batchsize + 16 * batchsize,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16 + 512,
            hidden_shape1,
            output_shape,
            rank1,
            2,
            8*8*20*20,
			20*20*20*32*32
        );
        
        relu_backward_inplace(
            data_input_output_buffer,
            28*28*batchsize + i * 512,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16,
            512
        );
        tensor_train_backward(
            data_input_output_buffer,
            weight[0],
            weight_grad[0],
            28*28*i,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16 + 512 + 20*20*20*32*32*4,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16 + 512,
            input_shape,
            hidden_shape1,
            rank0,
            4,
            8*8*20*20,
			20*20*20*32*32
        );
        for (int i = 0; i < 512; i++) {
            bias_grad[0][i] += data_input_output_buffer[28*28*batchsize + 
                512*batchsize + 16*batchsize + 16 + i];
        }
        for (int i = 0; i < 16; i++) {
            bias_grad[1][i] += data_input_output_buffer[28*28*batchsize + 
                512*batchsize + 16*batchsize+ i];
        }
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
    delete []buffer;
}

template<typename T>
void save_file(T* data, const char* filename, int size) {
    float* buffer = new float[size];
    for(int i = 0; i < size; i++) {
        buffer[i] = float(data[i]);
    }
    FILE* f = fopen(filename, "wb");
    fwrite((void*)buffer, sizeof(float), size, f);
    delete []buffer;
    fclose(f);
}

int main(){
    TYPE_DATA* data_input_output_buffer = new TYPE_DATA[32768000*5];
    TYPE_WEIGHT* weight[2];
    TYPE_DATA* bias[2];
    TYPE_DATA* weight_grad[2];
    TYPE_DATA* bias_grad[2];
    unsigned char label[100];
    assert(data_input_output_buffer != 0);
    weight[0] = new TYPE_WEIGHT[8*8*20*20*4];
    weight[1] = new TYPE_WEIGHT[8*8*20*20*2];
    weight_grad[0] = new TYPE_DATA[8*8*20*20*4];
    weight_grad[1] = new TYPE_DATA[8*8*20*20*2];
    bias[0] = new TYPE_DATA[512];
    bias[1] = new TYPE_DATA[16];
    bias_grad[0] = new TYPE_DATA[512];
    bias_grad[1] = new TYPE_DATA[16];

    load_file(weight[0], "weight0.bin", 7*4*16);
    load_file(weight[0] + 8*8*20*20, "weight1.bin", 16*4*8*16);
    load_file(weight[0] + 8*8*20*20*2, "weight2.bin", 16*7*4*16);
    load_file(weight[0] + 8*8*20*20*3, "weight3.bin", 16*4*4);
    load_file(weight[1], "weight4.bin", 32*4*16);
    load_file(weight[1] + 8*8*20*20, "weight5.bin", 16*4*16);
    load_file(bias[0], "bias0.bin", 512);
    load_file(bias[1], "bias1.bin", 16);
    load_file(data_input_output_buffer, "input.bin", 28*28*100);
    load_file(label, "label.bin", 100);

    mnist_train(data_input_output_buffer, weight, bias, weight_grad, bias_grad, label, 100);

    save_file(data_input_output_buffer + 28*28*100 + 512*100, "output.bin", 16*100);
    save_file(weight_grad[0], "weight_grad0.bin", 7*4*16);
    save_file(weight_grad[0] + 8*8*20*20, "weight_grad1.bin", 16*4*8*16);
    save_file(weight_grad[0] + 8*8*20*20*2, "weight_grad2.bin", 16*7*4*16);
    save_file(weight_grad[0] + 8*8*20*20*3, "weight_grad3.bin", 16*4*4);
    save_file(weight_grad[1], "weight_grad4.bin", 32*4*16);
    save_file(weight_grad[1] + 8*8*20*20, "weight_grad5.bin", 16*4*16);
    delete[] weight[0];
    delete[] weight[1];
    delete[] bias[0];
    delete[] bias[1];
    delete[] weight_grad[0];
    delete[] weight_grad[1];
    delete[] bias_grad[0];
    delete[] bias_grad[1];
    delete[] data_input_output_buffer;
}