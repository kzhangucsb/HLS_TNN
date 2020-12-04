#include "tt_nn.h"
#define WD (8*8*20*20)
#define TD (20*20*20*32*32)
#define IS (28*28)
#define HS 512
#define OS 16
#define BS 100
#define HO ((IS+OS)*BS)
#define TO ((IS+OS)*BS+HS+OS+HS)

void mnist_train_sync(
    TYPE_DATA* array,
    TYPE_WEIGHT* weight,
    TYPE_DATA* bias,
    TYPE_DATA* weight_grad,
    TYPE_DATA* bias_grad,
    unsigned char* label
){
    const int num_class = 16;
    for (int i_sample = 0; i_sample < 100; i_sample++) {
        tensor_cont_last(array, weight, i_sample * IS , 16465040, 76800, 196, 1, 4, 64);
        tensor_cont_mid(array, weight, 16465040, 8273040, 51200, 28, 112, 4, 64, 1);
        tensor_cont_mid(array, weight, 8273040, 81040, 25600, 7, 64, 16, 128, 1);
        tensor_cont_mid(array, weight, 81040, 80000, 0, 1, 112, 128, 4, 1);
        relu_inplace(array, 80000, 512);
        tensor_cont_last(array, weight, 80000, 81040, 153600, 32, 1, 16, 64);
        tensor_cont_mid(array, weight, 81040, 78400 + i_sample * OS, 128000, 1, 512, 4, 4, 1);
        softmax_ce_grad(array, label[i_sample], 78400 + i_sample * OS, 80512, num_class);
        tensor_cont_mid(array, weight, 80512, 81040, 128000, 1, 4, 4, 1, 512);
        tensor_cont_last(array, weight, 81040, 80528, 179200, 32, 16, 4, 16);
        tensor_cont_head_backward(array, weight_grad, 81040, 80000, 153600, 32, 64, 16);
        tensor_cont_last(array, weight, 80000, 81040, 153600, 32, 1, 16, 64);
        tensor_cont_end_backward(array, weight_grad, 81040, 80512, 128000, 1, 1, 512, 4, 4);
        relu_backward_inplace(array, 80000, 80528, 512);
        tensor_cont_mid(array, weight, 80528, 81040, 0, 1, 4, 128, 1, 112);
        tensor_cont_mid(array, weight, 81040, 8273040, 25600, 7, 128, 16, 1, 64);
        tensor_cont_mid(array, weight, 8273040, 16465040, 51200, 28, 64, 4, 1, 112);
        tensor_cont_last(array, weight, 16465040, 24657040, 102400, 196, 16, 4, 4);
        tensor_cont_end_backward(array, weight_grad, 16465040, 0, 76800, 196, 64, 4);
        tensor_cont_last(array, weight, 8273040, 16465040, 102400, 1792, 1, 4, 64);
        tensor_cont_end_backward(array, weight_grad, 16465040, 0, 51200, 28, 64, 16, 4, 7);
        tensor_cont_mid(array, weight, 81040, 8273040, 51200, 896, 4, 4, 16, 112);
        tensor_cont_last(array, weight, 8273040, 16465040, 102400, 100352, 16, 4, 4);
        tensor_cont_end_backward(array, weight_grad, 16465040, 0, 25600, 7, 128, 16, 28, 4);
        tensor_cont_last(array, weight, i_sample * IS, 16465040, 76800, 196, 1, 4, 64);
        tensor_cont_mid(array, weight, 16465040, 8273040, 51200, 28, 112, 4, 64, 1);
        tensor_cont_mid(array, weight, 8273040, 81040, 25600, 7, 64, 16, 128, 1);
        tensor_cont_end_backward(array, weight_grad, 81040, 80528, 0, 1, 1, 112, 128, 4);
        for (int i = 0; i < 512; i++) {
            bias_grad[i] += array[HO + HS + OS + i];
        }
        for (int i = 0; i < 16; i++) {
            bias_grad[HS + i] += array[HO + HS + i];
        }
    }
}

