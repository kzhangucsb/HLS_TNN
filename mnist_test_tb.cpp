#include <stdio.h>
#include <assert.h>
#include "tt_nn.h"

#define WD (8*8*20*20)
#define TD (20*20*20*32*32)
#define IS (28*28)
#define HS 512
#define OS 16
#define BS 100
#define HO ((IS+OS)*BS)
#define TO ((IS+OS)*BS+HS+OS+HS)

inline int sub2ind3(
    int ind0,
    int ind1,
    int ind2,
    int size1,
    int size2
){
#pragma HLS INLINE
    return (ind0 * size1 + ind1) * size2 + ind2;
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

/*********************************
 * data_input_output_buffer:
 * data_in*BS | output*BS | hidden | out_grad | hidden_grad | tmp (reused by sample)
 * forward: reused by layer
 * backward: output_grad | hidden_grad | tmp(reused by layer)
***********************************/


void mnist_forward(
    TYPE_DATA* data_input_output_buffer,
    TYPE_WEIGHT* weight,
    TYPE_DATA* bias
){
    
    int input_shape[] = {7, 4, 7, 4};
    int hidden_shape0[] = {4, 8, 4, 4};
    int rank0[] = {16, 16, 16};
    int hidden_shape1[] = {32, 16};
    
    int rank1[] = {16};
    int output_shape[] = {4, 4};

    for (int i = 0; i < BS; i++) {
        tensor_train_forward(
            data_input_output_buffer,
            weight,
            bias,
            input_shape,
            hidden_shape0,
            rank0,
            4,
            i * IS,
            HO,
            TO,
            0,
            0,
            TD,
            WD
        );
        relu_inplace(data_input_output_buffer, HO, HS);
        tensor_train_forward(
            data_input_output_buffer,
            weight,
            bias,
            hidden_shape1,
            output_shape,
            rank1,
            2,
            HO,
            IS * BS + i * 16,
            TO,
            WD * 4,
            HS,
			TD,
            WD
        );
    }
}

void mnist_train(
    TYPE_DATA* data_input_output_buffer,
    TYPE_WEIGHT* weight,
    TYPE_DATA* bias,
    TYPE_DATA* weight_grad,
    TYPE_DATA* bias_grad,
    unsigned char* label
){
    
    int input_shape[] = {7, 4, 7, 4};
    int hidden_shape0[] = {4, 8, 4, 4};
    int rank0[] = {16, 16, 16};
    int hidden_shape1[] = {32, 16};
    
    int rank1[] = {16};
    int output_shape[] = {4, 4};

    for (int i_0 = 0; i_0 < rank0[2]; i_0++) {
        for (int i_1 = 0; i_1 < hidden_shape0[3]; i_1++) {
            for (int i_2 = 0; i_2 < input_shape[3]; i_2++) {
                int ind_i = sub2ind3(i_0, i_1, i_2, hidden_shape0[3], input_shape[3]);
                int ind_o = sub2ind3(i_0, i_2, i_1, input_shape[3], hidden_shape0[3]);
                weight[4 * WD + ind_o] = weight[3 * WD + ind_i];
            }
        }
    }

    for (int i_0 = 0; i_0 < rank1[0]; i_0++) {
        for (int i_1 = 0; i_1 < output_shape[1]; i_1++) {
            for (int i_2 = 0; i_2 < hidden_shape1[1]; i_2++) {
                int ind_i = sub2ind3(i_0, i_1, i_2, output_shape[1], hidden_shape1[1]);
                int ind_o = sub2ind3(i_0, i_2, i_1, hidden_shape1[1], output_shape[1]);
                weight[7 * WD + ind_o] = weight[6 * WD + ind_i];
            }
        }
    }

    for (int i = 0; i < 1; i++) {
        tensor_train_forward(
            data_input_output_buffer,
            weight,
            bias,
            input_shape,
            hidden_shape0,
            rank0,
            4,
            i * IS,
            HO,
            TO,
            0,
            0,
            TD,
            WD
        );
        relu_inplace(data_input_output_buffer, HO, HS);
        tensor_train_forward(
            data_input_output_buffer,
            weight,
            bias,
            hidden_shape1,
            output_shape,
            rank1,
            2,
            HO,
            IS * BS + i * OS,
            TO,
            WD * 5,
            HS,
			TD,
            WD
        );
        // backward
        softmax_ce_grad(
            data_input_output_buffer, 
            label[i], 
            IS * BS + i * OS,
            HO + HS,
            OS
        );
        tensor_train_backward(
            data_input_output_buffer,
            weight,
            weight_grad,
            hidden_shape1,
            output_shape,
            rank1,
            2,
            HO,
            HO + HS,
            HO + HS + OS,
            TO,
            WD * 5,
            TD,
            WD
        );
        
        relu_backward_inplace(
            data_input_output_buffer,
            HO,
            HO + HS + OS,
            HS
        );
        tensor_train_backward(
            data_input_output_buffer,
            weight,
            weight_grad,
            input_shape,
            hidden_shape0,
            rank0,
            4,
            IS * i,
            HO + HS + OS, 
            TO + TD * 3,
            TO,
            0,
            TD,
            WD
        );
        for (int i = 0; i < 512; i++) {
            bias_grad[i] += data_input_output_buffer[HO + HS + OS + i];
        }
        for (int i = 0; i < 16; i++) {
            bias_grad[HS + i] += data_input_output_buffer[HO + HS + i];
        }
    }
}



int main(){
    TYPE_DATA* data_input_output_buffer = new TYPE_DATA[32768000*5];
    TYPE_WEIGHT* weight;
    TYPE_DATA* bias;
    TYPE_DATA* weight_grad;
    TYPE_DATA* bias_grad;
    unsigned char label[100];
    assert(data_input_output_buffer != 0);
    weight = new TYPE_WEIGHT[WD * 8];
    weight_grad = new TYPE_DATA[WD * 8];
    bias = new TYPE_DATA[HS + OS];
    bias_grad = new TYPE_DATA[HS + OS];

    load_file(weight, "weight0.bin", 7*4*16);
    load_file(weight + WD, "weight1.bin", 16*4*8*16);
    load_file(weight + WD * 2, "weight2.bin", 16*7*4*16);
    load_file(weight + WD * 3, "weight3.bin", 16*4*4);
    load_file(weight + WD * 5, "weight4.bin", 32*4*16);
    load_file(weight + WD * 6, "weight5.bin", 16*4*16);
    load_file(bias, "bias0.bin", 512);
    load_file(bias + HS, "bias1.bin", 16);
    load_file(data_input_output_buffer, "input.bin", 28*28*100);
    load_file(label, "label.bin", 100);

    mnist_train(data_input_output_buffer, weight, bias, weight_grad, bias_grad, label);

    save_file(data_input_output_buffer + IS * BS, "output.bin", 16*100);
    save_file(weight_grad, "weight_grad0.bin", 7*4*16);
    save_file(weight_grad + WD, "weight_grad1.bin", 16*4*8*16);
    save_file(weight_grad + WD * 2, "weight_grad2.bin", 16*7*4*16);
    save_file(weight_grad + WD * 3, "weight_grad3.bin", 16*4*4);
    save_file(weight_grad + WD * 5, "weight_grad4.bin", 32*4*16);
    save_file(weight_grad + WD * 6, "weight_grad5.bin", 16*4*16);
    save_file(bias_grad, "bias_grad0.bin", 512);
    save_file(bias_grad + HS, "bias_grad1.bin", 16);
    delete[] weight;
    delete[] bias;
    delete[] weight_grad;
    delete[] bias_grad;
    delete[] data_input_output_buffer;
}
