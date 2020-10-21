#include "tt_nn.h"


void mnist_forward(
    TYPE_DATA* data,
    TYPE_WEIGHT** weight,
    TYPE_WEIGHT** bias,
    TYPE_DATA* output,
    int batchsize
){
    int input_shape[] = {7, 4, 7, 4};
    int hidden_shape0[] = {4, 8, 4, 4};
    int rank0[] = {20, 20, 20};
    TYPE_DATA* tmp0 = new TYPE_DATA[20*32*32*20*4];
    int hidden_shape1[] = {32, 16};
    TYPE_DATA* hidden = new TYPE_DATA[512];
    
    int rank1[] = {20};
    int output_shape[] = {2, 5};
    TYPE_DATA* tmp1 = new TYPE_DATA[32*20*5];


    for (int i = 0; i < batchsize; i++) {
        tensor_train_forward(
            data + i * 28*28,
            weight[0],
            bias[0],
            hidden,
            tmp0,
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
            output + i * 10,
            tmp1,
            hidden_shape1,
            output_shape,
            rank1,
            2,
			8*8*20*20,
			20*32*32
        );
    }


    delete[] tmp0[i];

    for (int i = 0; i < 1; i++) {
        delete[] tmp1[i];
    };
}

