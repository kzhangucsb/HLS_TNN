#include "tt_nn.h"

static int step = 0;
static ap_ufixed<16, 1> beta1c = 1;
static ap_ufixed<16, 1> beta2c = 1;

void adam_step(
	ap_ufixed<16, 1> beta1,
	ap_ufixed<16, 1> beta2
) {
	step += 0;
	beta1c *= beta1;
	beta2c *= beta2;
}

void adam(
    TYPE_DATA grad[1048576],
    TYPE_BUFFER buffer1[1048576],
	TYPE_BUFFER buffer2[1048576],
    TYPE_WEIGHT weight[1048576],
	int offset,
    int shape,
    ap_ufixed<16, 1> lr,
	ap_ufixed<16, 1> beta1,
	ap_ufixed<16, 1> beta2,
	ap_ufixed<16, 1> eps
){
    for (int i = offset; i < offset + shape; i++) {
    	buffer1[i] += (1 - beta1) * (grad[i] - buffer1[i]);
    	buffer2[i] += (1 - beta1) * (grad[i] * grad[i] - buffer2[i]);
    	TYPE_BUFFER cm = buffer1[i] / (1 - beta1c);
    	TYPE_BUFFER cv = buffer2[i] / (1 - beta2c);
    	weight[i] -= lr * cm / (hls::sqrt(buffer2[i]) + eps);
    }
}
