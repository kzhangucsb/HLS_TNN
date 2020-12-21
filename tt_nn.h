#define SYNTHESIS
#ifdef SYNTHESIS
#include <ap_fixed.h>
#include <hls_math.h>
#define exp hls::exp
#define TYPE_WEIGHT ap_fixed<4, 1>
#define TYPE_DATA ap_fixed<8, 1>
#define TYPE_GRAD ap_fixed<16, 9>
#define TYPE_INTER ap_fixed<16, 1>
#define TYPE_WEIGHT_BUFF ap_fixed<8, 1>
#define TYPE_BUFFER ap_fixed<32, 1>
#else
#include <math.h>
typedef float TYPE_WEIGHT ;
typedef float TYPE_DATA;
typedef float TYPE_INTER;
typedef float TYPE_BUFFER;
#endif

#define PARALLEL_DEGREE 8

void tensor_cont_mid(
    TYPE_DATA array[1073741824],
    TYPE_WEIGHT weight[1048576],
    int in_offset,
    int out_offset,
    int weight_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2,
	int shift
);

void tensor_cont_last(
    TYPE_DATA array[1073741824],
    TYPE_WEIGHT weight[1048576],
    int in_offset,
    int out_offset,
    int weight_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_1,
	int shift
);

void tensor_cont_end_backward(
    TYPE_DATA array[1073741824],
    TYPE_GRAD grad_out[1073741824],
    int a1_offset,
    int a2_offset,
    int out_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_in_size_3,
    int array_weight_size_1,
	int shift
);

void tensor_cont_head_backward(
    TYPE_DATA array[1073741824],
    TYPE_GRAD grad_out[1073741824],
    int a1_offset,
    int a2_offset,
    int out_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_weight_size_1,
	int shift
);

void tensor_train_forward(
    TYPE_DATA array_list[1073741824],
    TYPE_WEIGHT weight[1048576],
    TYPE_DATA bias[1048576],
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
    int array_in_offset,
	int array_out_offset,
	int tmp_offset,
	int weight_offset,
    int bias_offset,
	int tmp_distance,
    int weight_distance
);

void tensor_train_input_grad(
    TYPE_DATA array_list[1073741824],
    TYPE_WEIGHT weight[1048576],
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
    int grad_out_offset,
    int grad_in_offset,
	int tmp_offset,
	int weight_offset,
	int tmp_distance,
    int weight_distance
);

void tensor_train_weight_grad(
    TYPE_DATA array_list[1073741824],
    TYPE_GRAD weight[1048576],
    TYPE_GRAD weight_grad[1048576],
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
    int dim_grad,
    int array_in_offset,
    int grad_out_offset,
	int tmp_offset,
	int weight_offset,
	int tmp_distance,
    int weight_distance
);

void tensor_train_backward(
    TYPE_DATA array_list[1073741824],
    TYPE_WEIGHT weight[1048576],
    TYPE_GRAD weight_grad[1048576],
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
    int array_in_offset,
    int grad_out_offset,
    int grad_in_offset,
	int tmp_offset,
	int weight_offset,
	int tmp_distance,
    int weight_distance
);

void relu_inplace(
    TYPE_DATA* data,
    int offset,
    int shape
);

void relu_backward_inplace(
    TYPE_DATA* data,
    int data_offset,
    int grad_offset,
    int shape
);

void softmax_ce_grad(
    TYPE_DATA* data,
    unsigned char label,
    int out_offset,
    int grad_offset,
    unsigned char num_class
);

void adam_step(
	ap_ufixed<16, 1> beta1,
	ap_ufixed<16, 1> beta2
);

void adam(
    TYPE_GRAD grad[1048576],
    TYPE_BUFFER buffer1[1048576],
	TYPE_BUFFER buffer2[1048576],
    TYPE_WEIGHT weight[1048576],
	TYPE_WEIGHT_BUFF weight_buffer[1048576],
	int offset,
    int shape,
    ap_ufixed<16, 1> lr,
	ap_ufixed<16, 1> beta1,
	ap_ufixed<16, 1> beta2,
	ap_ufixed<16, 1> eps
);

void get_rank_para_update(
	TYPE_WEIGHT_BUFF weight_buffer[1048576],
	float rank_parameter[1048576],
	int offset,
	int num_rank,
	int num_para_per_rank,
	float em_stepsize
);

void add_bayes_grad(
	TYPE_WEIGHT_BUFF weight_buffer[1048576],
	TYPE_GRAD grad[1048576],
	float rank_parameter[1048576],
	float scale,
	int offset,
	int num_rank,
	int num_para_per_rank
);







