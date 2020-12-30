#include "tt_nn.h"




#define WD (16*16*16)
#define TD (20*20*20*32*32)
#define IS (28*28)
#define HS 512
#define OS 16
#define BS 100
#define HO ((IS+OS)*BS)
#define TO ((IS+OS)*BS+HS+OS+HS)
#define LR 0.001
#define BETA1 0.9
#define BETA2 0.999
#define EPS 0.001

void mnist_train_sync(
    TYPE_DATA array[1073741824],
	TYPE_DATA array2[1073741824],
    //TYPE_WEIGHT weight[1048576],
    //TYPE_DATA bias[1048576],
    //TYPE_DATA weight_grad[1048576],
    //TYPE_DATA bias_grad[1048576],
    unsigned char label[1048576],
	TYPE_GRAD weight_grad[WD*6],
	TYPE_GRAD bias_grad[HS+OS],
	int shift
){
	TYPE_WEIGHT weight[WD*12];
	TYPE_DATA bias[HS+OS];
	
	const int stride1[] = {896, 128, 32, 16, 1};
	const int stride2[] = {512, 128, 32, 16, 1};
	const int strideo[] = {458752, 458752, 114688, 16384, 4096, 1024, 512, 256, 16, 1};
	const int size1[] = {1, 7, 4, 2, 16};
	const int size2[] = {1, 4, 4, 2, 16};

	// TYPE_BUFFER weight_buffer1[WD*8];
	// TYPE_BUFFER weight_buffer2[WD*8];
	// TYPE_WEIGHT_BUFF weight_buffer3[WD*8];
	// float rank_parameters0[16];
	// TYPE_BUFFER bias_buffer1[HS+OS];
	// TYPE_BUFFER bias_buffer2[HS+OS];
#pragma HLS INTERFACE m_axi depth=1073741824 port=array offset=slave
#pragma HLS INTERFACE m_axi depth=1073741824 port=array2 offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
#pragma HLS INTERFACE ap_memory depth=1048576 port=bias
#pragma HLS INTERFACE m_axi depth=1048576 port=weight_grad
#pragma HLS INTERFACE m_axi depth=1048576 port=bias_grad
#pragma HLS ARRAY_RESHAPE variable=weight cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=bias cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=weight_grad cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=bias_grad cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=array cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=array2 cyclic factor=16 dim=1
    const int num_class = 16;
    for (int i_sample = 0; i_sample < 100; i_sample++) {
    	tensor_cont_last(array, array2, weight, i_sample * IS, 16465040, 76800, 56, 1, 16, 256, shift);
    	tensor_cont_mid(array2, array, weight, 16465040, 8273040, 51200, 28, 32, 16, 32, 1, shift);
    	tensor_cont_mid(array, array2, weight, 8273040, 81040, 25600, 7, 64, 32, 64, 1, shift);
    	tensor_cont_mid(array2, array, weight, 81040, 80000, 0, 1, 112, 128, 4, 1, shift);
    	relu_inplace(array, 80000, 512);
    	tensor_cont_last(array, array2, weight, 80000, 81040, 153600, 32, 1, 16, 256, shift);
    	tensor_cont_mid(array2, array, weight, 81040, 78400, 128000, 1, 512, 16, 1, 1, shift);
    	softmax_ce_grad(array, label[i_sample], 78400, 80512, num_class);
    	tensor_cont_mid(array, array2, weight, 80512, 81040, 128000, 1, 1, 16, 1, 512, shift);
    	tensor_cont_last(array, array2, weight, 81040, 80528, 179200, 32, 16, 16, 16, shift);
    	relu_backward_inplace(array, 80000, 80528, 512);
    	tensor_cont_mid(array, array2, weight, 80528, 81040, 0, 1, 4, 128, 1, 112, shift);
    	tensor_cont_mid(array2, array, weight, 81040, 8273040, 25600, 7, 64, 32, 1, 64, shift);
    	tensor_cont_mid(array, array2, weight, 8273040, 16465040, 51200, 28, 32, 16, 1, 32, shift);
    	tensor_cont_last(array2, array, weight, 16465040, 24657040, 102400, 56, 16, 16, 16, shift);
    	tensor_cont_outer_prod(array, array2, i_sample * IS, 80528, 24657040,
    			stride1, stride2, strideo, size1, size2, shift);


        for (int i = 0; i < 512; i++) {
            bias_grad[i] += array[HO + HS + OS + i];
        }
        for (int i = 0; i < 16; i++) {
            bias_grad[HS + i] += array[HO + HS + i];
        }
        // get_rank_para_update(
        // 	weight_buffer3,
        // 	rank_parameters0,
        // 	0,
        // 	1,
        // 	7*4*16,
        // 	0.1
        // );
        // add_bayes_grad(
        // 	weight_buffer3,
		// 	weight_grad,
        // 	rank_parameters0,
        // 	0.1,
        // 	0,
        // 	1,
        // 	7*4*16
        // );

        // adam_step(BETA1, BETA2);
        // adam(weight_grad, weight_buffer1, weight_buffer2, weight, weight_buffer3, 0, 7*4*16, LR, BETA1, BETA2, EPS);

    }
}


