#define SYNTHESIS
#ifdef SYNTHESIS
#include <ap_fixed.h>
#include <hls_math.h>
#define exp hls::exp
#define TYPE_WEIGHT ap_fixed<8, 0>
#define TYPE_DATA ap_fixed<8, 0>
#define TYPE_INTER ap_fixed<16, 0>
#else
#include <math.h>
typedef float TYPE_WEIGHT ;
typedef float TYPE_DATA;
typedef float TYPE_INTER;
#endif

#define PARALLEL_DEGREE 4

void tensor_cont_mid(
    TYPE_DATA array_in[1073741824],
    TYPE_WEIGHT array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2
);

void tensor_cont_last(
    TYPE_DATA array_in[1073741824],
    TYPE_WEIGHT array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2
);

void tensor_train_forward(
    TYPE_DATA array_list[1073741824],
    TYPE_WEIGHT weight[1048576],
    TYPE_DATA bias[1048576],
    int array_in_offset,
	int array_out_offset,
	int tmp_offset,
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
	int weight_offset,
	int tmp_distance
);

void tensor_cont_end_backward(
    TYPE_DATA array_in[1073741824],
    TYPE_DATA array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_in_size_3,
    int array_weight_size_1
);

void tensor_cont_head_backward(
    TYPE_DATA array_in[1073741824],
    TYPE_DATA array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_weight_size_1
);

void tensor_train_input_grad(
    TYPE_DATA array_list[1073741824],
    TYPE_WEIGHT weight[1048576],
	int grad_out_offset,
    int grad_in_offset,
	int tmp_offset,
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
	int weight_offset,
	int tmp_distance
);

void tensor_train_weight_grad(
    TYPE_DATA array_list[1073741824],
    TYPE_WEIGHT weight[1048576],
    TYPE_DATA weight_grad[1048576],
    int dim_grad,
    int array_in_offset,
    int grad_out_offset,
	int tmp_offset,
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
	int weight_offset,
	int tmp_distance
);

void tensor_train_backward(
    TYPE_DATA array_list[1073741824],
    TYPE_WEIGHT weight[1048576],
    TYPE_DATA weight_grad[1048576],
    int array_in_offset,
    int grad_out_offset,
    int grad_in_offset,
	int tmp_offset,
    int input_shape[4],
    int output_shape[4],
    int rank[4],
    int dim,
	int weight_offset,
	int tmp_distance
);

void relu_inplace(
    TYPE_DATA* data,
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
