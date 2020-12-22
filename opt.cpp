#include "tt_nn.h"

static int step = 0;
static ap_ufixed<16, 1> beta1c = 1;
static ap_ufixed<16, 1> beta2c = 1;

void adam_step(
	TYPE_PARA beta1,
	TYPE_PARA beta2
) {
	step += 0;
	beta1c *= beta1;
	beta2c *= beta2;
}

void adam(
    TYPE_GRAD grad[1048576],
    TYPE_BUFFER buffer1[1048576],
	TYPE_BUFFER buffer2[1048576],
    TYPE_WEIGHT weight[1048576],
	TYPE_WEIGHT_BUFF weight_buffer[1048576],
	int offset,
    int shape,
	TYPE_PARA lr,
	TYPE_PARA beta1,
	TYPE_PARA beta2,
	TYPE_PARA eps
){
    for (int i = offset; i < offset + shape; i++) {
    	buffer1[i] += (1 - beta1) * (grad[i] - buffer1[i]);
    	buffer2[i] += (1 - beta1) * (grad[i] * grad[i] - buffer2[i]);
    	TYPE_BUFFER cm = buffer1[i] / (1 - beta1c);
    	TYPE_BUFFER cv = buffer2[i] / (1 - beta2c);
    	weight_buffer[i] -= lr * cm / (hls::sqrt(buffer2[i]) + eps);
    	weight[i] = weight_buffer[i];
    }
}

void get_rank_para_update(
	TYPE_WEIGHT_BUFF weight_buffer[1048576],
	float rank_parameter[1048576],
	int offset,
	int num_rank,
	int num_para_per_rank,
	float em_stepsize
){
	for (int i = 0; i < num_rank; i++) {
		float norm = 0;
		int offset_rank = i * num_para_per_rank;
		for (int j = 0; j < num_para_per_rank; j++) {
			norm += float(weight_buffer[offset + offset_rank + j] * weight_buffer[offset + offset_rank + j]);
		}
		rank_parameter[i] += em_stepsize * (norm - rank_parameter[i]);
	}
}

void add_bayes_grad(
	TYPE_WEIGHT_BUFF weight_buffer[1048576],
	TYPE_GRAD grad[1048576],
	float rank_parameter[1048576],
	TYPE_PARA scale,
	int offset,
	int num_rank,
	int num_para_per_rank
){
	for (int i = 0; i < num_rank; i++) {
		int offset_rank = i * num_para_per_rank;
		if (rank_parameter[i] > 0.001) {
			for (int j = 0; j < num_para_per_rank; j++) {
				grad[offset + offset_rank + j] += TYPE_GRAD(
						float(weight_buffer[offset + offset_rank + j]) / (rank_parameter[i])) * scale;
			}
		}
		else{
			for (int j = 0; j < num_para_per_rank; j++) {
				grad[offset + offset_rank + j] += TYPE_GRAD(
						float(weight_buffer[offset + offset_rank + j]) * 1000) * scale;
			}
		}
	}
}

