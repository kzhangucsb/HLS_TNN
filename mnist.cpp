#include "tt_nn.h"

void mnist_train_sync(
    TYPE_DATA data_input_output_buffer[1073741824],
    TYPE_WEIGHT weight0[1048576],
	TYPE_WEIGHT weight1[1048576],
    TYPE_DATA bias0[1048576],
	TYPE_DATA bias1[1048576],
    TYPE_DATA weight_grad0[1048576],
	TYPE_DATA weight_grad1[1048576],
    TYPE_DATA bias_grad0[1048576],
	TYPE_DATA bias_grad1[1048576],
    unsigned char label[1048576],
    int batchsize
){
#pragma HLS INTERFACE m_axi depth=1073741824 port=data_input_output_buffer offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight0
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight1
#pragma HLS INTERFACE ap_memory depth=1048576 port=bias0
#pragma HLS INTERFACE ap_memory depth=1048576 port=bias1
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight_grad0
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight_grad1
#pragma HLS INTERFACE ap_memory depth=1048576 port=bias_grad0
#pragma HLS INTERFACE ap_memory depth=1048576 port=bias_grad1
#pragma HLS ARRAY_PARTITION variable=weight0 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=weight1 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=bias0 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=bias1 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=weight_grad0 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=weight_grad1 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=bias_grad0 cyclic factor=16 dim=1
#pragma HLS ARRAY_PARTITION variable=bias_grad1 cyclic factor=16 dim=1

    int input_shape[] = {7, 4, 7, 4};
    int hidden_shape0[] = {4, 8, 4, 4};
    int rank0[] = {16, 16, 16};
    int hidden_shape1[] = {32, 16};

    int rank1[] = {16};
    int output_shape[] = {4, 4};

    for (int i = 0; i < 1; i++) {
        tensor_train_forward(
            data_input_output_buffer,
            weight0,
            bias0,
            i * 28*28,
            28*28*batchsize + 512 * i,
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
            weight1,
            bias1,
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
            weight1,
            weight_grad1,
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
            weight0,
            weight_grad0,
            28*28*i,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16 + 512 + 20*20*20*32*32*4,
            28*28*batchsize + 512*batchsize + 16 * batchsize + 16 + 512,
            input_shape,
            hidden_shape0,
            rank0,
            4,
            8*8*20*20,
			20*20*20*32*32
        );
        for (int i = 0; i < 512; i++) {
            bias_grad0[i] += data_input_output_buffer[28*28*batchsize +
                512*batchsize + 16*batchsize + 16 + i];
        }
        for (int i = 0; i < 16; i++) {
            bias_grad1[i] += data_input_output_buffer[28*28*batchsize +
                512*batchsize + 16*batchsize+ i];
        }
    }
}
