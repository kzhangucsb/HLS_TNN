#include "tt_nn.h"


//void mnist_train_sync(
//    TYPE_DATA data_input_output_buffer[1073741824],
//    TYPE_WEIGHT weight0[1048576],
//	TYPE_WEIGHT weight1[1048576],
//    TYPE_DATA bias0[1048576],
//	TYPE_DATA bias1[1048576],
//    TYPE_DATA weight_grad0[1048576],
//	TYPE_DATA weight_grad1[1048576],
//    TYPE_DATA bias_grad0[1048576],
//	TYPE_DATA bias_grad1[1048576],
//    unsigned char label[1048576],
//    int batchsize
//){
//#pragma HLS INTERFACE m_axi depth=1073741824 port=data_input_output_buffer offset=slave
//#pragma HLS INTERFACE ap_memory depth=1048576 port=weight0
//#pragma HLS INTERFACE ap_memory depth=1048576 port=weight1
//#pragma HLS INTERFACE ap_memory depth=1048576 port=bias0
//#pragma HLS INTERFACE ap_memory depth=1048576 port=bias1
//#pragma HLS INTERFACE ap_memory depth=1048576 port=weight_grad0
//#pragma HLS INTERFACE ap_memory depth=1048576 port=weight_grad1
//#pragma HLS INTERFACE ap_memory depth=1048576 port=bias_grad0
//#pragma HLS INTERFACE ap_memory depth=1048576 port=bias_grad1
//#pragma HLS ARRAY_RESHAPE variable=weight0 cyclic factor=16 dim=1
//#pragma HLS ARRAY_RESHAPE variable=weight1 cyclic factor=16 dim=1
//#pragma HLS ARRAY_RESHAPE variable=bias0 cyclic factor=16 dim=1
//#pragma HLS ARRAY_RESHAPE variable=bias1 cyclic factor=16 dim=1
//#pragma HLS ARRAY_RESHAPE variable=weight_grad0 cyclic factor=16 dim=1
//#pragma HLS ARRAY_RESHAPE variable=weight_grad1 cyclic factor=16 dim=1
//#pragma HLS ARRAY_RESHAPE variable=bias_grad0 cyclic factor=16 dim=1
//#pragma HLS ARRAY_RESHAPE variable=bias_grad1 cyclic factor=16 dim=1
//#pragma HLS ARRAY_RESHAPE variable=data_input_output_buffer cyclic factor=16 dim=1
//
//    int input_shape[] = {7, 4, 7, 4};
//    int hidden_shape0[] = {4, 8, 4, 4};
//    int rank0[] = {16, 16, 16};
//    int hidden_shape1[] = {32, 16};
//
//    int rank1[] = {16};
//    int output_shape[] = {4, 4};
//
//    for (int i = 0; i < 1; i++) {
//        tensor_train_forward(
//            data_input_output_buffer,
//            weight0,
//            bias0,
//            i * 28*28,
//            28*28*batchsize + 512 * i,
//            28*28*batchsize + 512*batchsize + 16 * batchsize,
//            input_shape,
//            hidden_shape0,
//            rank0,
//            4,
//			8*8*20*20,
//			20*32*32
//        );
//        relu_inplace(data_input_output_buffer + 28*28*batchsize + i * 512, 512);
//        tensor_train_forward(
//            data_input_output_buffer,
//            weight1,
//            bias1,
//            28*28*batchsize + i * 512,
//            28*28*batchsize + 512*batchsize + i * 16,
//            28*28*batchsize + 512*batchsize + 16 * batchsize,
//            hidden_shape1,
//            output_shape,
//            rank1,
//            2,
//			8*8*20*20,
//			20*32*32
//        );
//        // backward
//        softmax_ce_grad(
//            data_input_output_buffer,
//            label[i],
//            28*28*batchsize + 512*batchsize + i * 16,
//            28*28*batchsize + 512*batchsize + 16 * batchsize,
//            16
//        );
//        tensor_train_backward(
//            data_input_output_buffer,
//            weight1,
//            weight_grad1,
//            28*28*batchsize + 512 * i,
//            28*28*batchsize + 512*batchsize + 16 * batchsize,
//            28*28*batchsize + 512*batchsize + 16 * batchsize + 16,
//            28*28*batchsize + 512*batchsize + 16 * batchsize + 16 + 512,
//            hidden_shape1,
//            output_shape,
//            rank1,
//            2,
//            8*8*20*20,
//			20*20*20*32*32
//        );
//
//        relu_backward_inplace(
//            data_input_output_buffer,
//            28*28*batchsize + i * 512,
//            28*28*batchsize + 512*batchsize + 16 * batchsize + 16,
//            512
//        );
//        tensor_train_backward(
//            data_input_output_buffer,
//            weight0,
//            weight_grad0,
//            28*28*i,
//            28*28*batchsize + 512*batchsize + 16 * batchsize + 16,
//            28*28*batchsize + 512*batchsize + 16 * batchsize + 16 + 512 + 20*20*20*32*32*4,
//            28*28*batchsize + 512*batchsize + 16 * batchsize + 16 + 512,
//            input_shape,
//            hidden_shape0,
//            rank0,
//            4,
//            8*8*20*20,
//			20*20*20*32*32
//        );
//        for (int i = 0; i < 512; i++) {
//            bias_grad0[i] += data_input_output_buffer[28*28*batchsize +
//                512*batchsize + 16*batchsize + 16 + i];
//        }
//        for (int i = 0; i < 16; i++) {
//            bias_grad1[i] += data_input_output_buffer[28*28*batchsize +
//                512*batchsize + 16*batchsize+ i];
//        }
//    }
//}

#define WD (8*8*20*20)
#define TD (20*20*20*32*32)
#define IS (28*28)
#define HS 512
#define OS 16
#define BS 100
#define HO ((IS+OS)*BS)
#define TO ((IS+OS)*BS+HS+OS+HS)

void mnist_train_sync(
    TYPE_DATA array[1073741824],
    TYPE_WEIGHT weight[1048576],
    TYPE_DATA bias[1048576],
    TYPE_DATA weight_grad[1048576],
    TYPE_DATA bias_grad[1048576],
    unsigned char label[1048576]
){
#pragma HLS INTERFACE m_axi depth=1073741824 port=array offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
#pragma HLS INTERFACE ap_memory depth=1048576 port=bias
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight_grad
#pragma HLS INTERFACE ap_memory depth=1048576 port=bias_grad
#pragma HLS ARRAY_RESHAPE variable=weight cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=bias cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=weight_grad cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=bias_grad cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=array cyclic factor=16 dim=1
    const int num_class = 16;
    for (int i_sample = 0; i_sample < 100; i_sample++) {
    	tensor_cont_last(array, weight, i_sample * IS, 16465040, 76800, 56, 1, 16, 256);
    	tensor_cont_mid(array, weight, 16465040, 8273040, 51200, 28, 32, 16, 32, 1);
    	tensor_cont_mid(array, weight, 8273040, 81040, 25600, 7, 64, 32, 64, 1);
    	tensor_cont_mid(array, weight, 81040, 80000, 0, 1, 112, 128, 4, 1);
    	relu_inplace(array, 80000, 512);
    	tensor_cont_last(array, weight, 80000, 81040, 153600, 32, 1, 16, 256);
    	tensor_cont_mid(array, weight, 81040, 78400, 128000, 1, 512, 16, 1, 1);
    	softmax_ce_grad(array, label[i_sample], 78400, 80512, num_class);
    	tensor_cont_mid(array, weight, 80512, 81040, 128000, 1, 1, 16, 1, 512);
    	tensor_cont_last(array, weight, 81040, 80528, 179200, 32, 16, 16, 16);
    	tensor_cont_head_backward(array, weight_grad, 81040, 80000, 153600, 32, 256, 16);
    	tensor_cont_last(array, weight, 80000, 81040, 153600, 32, 1, 16, 256);
    	tensor_cont_end_backward(array, weight_grad, 81040, 80512, 128000, 1, 1, 512, 16, 1);
    	relu_backward_inplace(array, 80000, 80528, 512);
    	tensor_cont_mid(array, weight, 80528, 81040, 0, 1, 4, 128, 1, 112);
    	tensor_cont_mid(array, weight, 81040, 8273040, 25600, 7, 64, 32, 1, 64);
    	tensor_cont_mid(array, weight, 8273040, 16465040, 51200, 28, 32, 16, 1, 32);
    	tensor_cont_last(array, weight, 16465040, 24657040, 102400, 56, 16, 16, 16);
    	tensor_cont_head_backward(array, weight_grad, 16465040, 0, 76800, 56, 256, 16);
    	tensor_cont_last(array, weight, 8273040, 16465040, 102400, 896, 1, 16, 256);
    	tensor_cont_end_backward(array, weight_grad, 16465040, i_sample * IS, 51200, 28, 32, 16, 16, 2);
    	tensor_cont_mid(array, weight, 81040, 8273040, 51200, 448, 2, 16, 16, 32);
    	tensor_cont_last(array, weight, 8273040, 16465040, 102400, 14336, 16, 16, 16);
    	tensor_cont_end_backward(array, weight_grad, 16465040, i_sample * IS, 25600, 7, 64, 16, 32, 4);
    	tensor_cont_last(array, weight, i_sample * IS, 16465040, 76800, 56, 1, 16, 256);
    	tensor_cont_mid(array, weight, 16465040, 8273040, 51200, 28, 32, 16, 32, 1);
    	tensor_cont_mid(array, weight, 8273040, 81040, 25600, 7, 64, 32, 64, 1);
    	tensor_cont_end_backward(array, weight_grad, 81040, 80528, 0, 1, 1, 112, 128, 4);
//        tensor_cont_last(array, weight, i_sample * IS , 16465040, 76800, 196, 1, 4, 64);
//        tensor_cont_mid(array, weight, 16465040, 8273040, 51200, 28, 112, 4, 64, 1);
//        tensor_cont_mid(array, weight, 8273040, 81040, 25600, 7, 64, 16, 128, 1);
//        tensor_cont_mid(array, weight, 81040, 80000, 0, 1, 112, 128, 4, 1);
//        relu_inplace(array, 80000, 512);
//        tensor_cont_last(array, weight, 80000, 81040, 153600, 32, 1, 16, 64);
//        tensor_cont_mid(array, weight, 81040, 78400 + i_sample * OS, 128000, 1, 512, 4, 4, 1);
//        softmax_ce_grad(array, label[i_sample], 78400 + i_sample * OS, 80512, num_class);
//        tensor_cont_mid(array, weight, 80512, 81040, 128000, 1, 4, 4, 1, 512);
//        tensor_cont_last(array, weight, 81040, 80528, 179200, 32, 16, 4, 16);
//        tensor_cont_head_backward(array, weight_grad, 81040, 80000, 153600, 32, 64, 16);
//        tensor_cont_last(array, weight, 80000, 81040, 153600, 32, 1, 16, 64);
//        tensor_cont_end_backward(array, weight_grad, 81040, 80512, 128000, 1, 1, 512, 4, 4);
//        relu_backward_inplace(array, 80000, 80528, 512);
//        tensor_cont_mid(array, weight, 80528, 81040, 0, 1, 4, 128, 1, 112);
//        tensor_cont_mid(array, weight, 81040, 8273040, 25600, 7, 128, 16, 1, 64);
//        tensor_cont_mid(array, weight, 8273040, 16465040, 51200, 28, 64, 4, 1, 112);
//        tensor_cont_last(array, weight, 16465040, 24657040, 102400, 196, 16, 4, 4);
//        tensor_cont_head_backward(array, weight_grad, 16465040, i_sample * IS, 76800, 196, 64, 4);
//        tensor_cont_last(array, weight, 8273040, 16465040, 102400, 1792, 1, 4, 64);
//        tensor_cont_end_backward(array, weight_grad, 16465040, i_sample * IS, 51200, 28, 64, 16, 4, 7);
//        tensor_cont_mid(array, weight, 81040, 8273040, 51200, 896, 4, 4, 16, 112);
//        tensor_cont_last(array, weight, 8273040, 16465040, 102400, 100352, 16, 4, 4);
//        tensor_cont_end_backward(array, weight_grad, 16465040, i_sample * IS, 25600, 7, 128, 16, 28, 4);
//        tensor_cont_last(array, weight, i_sample * IS, 16465040, 76800, 196, 1, 4, 64);
//        tensor_cont_mid(array, weight, 16465040, 8273040, 51200, 28, 112, 4, 64, 1);
//        tensor_cont_mid(array, weight, 8273040, 81040, 25600, 7, 64, 16, 128, 1);
//        tensor_cont_end_backward(array, weight_grad, 81040, 80528, 0, 1, 1, 112, 128, 4);
        for (int i = 0; i < 512; i++) {
            bias_grad[i] += array[HO + HS + OS + i];
        }
        for (int i = 0; i < 16; i++) {
            bias_grad[HS + i] += array[HO + HS + i];
        }
    }
}


