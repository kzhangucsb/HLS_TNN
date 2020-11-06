#ifdef SYNTHESIS
#include <ap_fixed.h>
#define TYPE_WEIGHT ap_fixed<8, 0>
#define TYPE_DATA ap_fixed<8, 0>
#define TYPE_INTER ap_fixed<16, 0>
#else
#define TYPE_WEIGHT float
#define TYPE_DATA float
#define TYPE_INTER float
#endif

#define PARALLEL_DEGREE 4

void tensor_contraction_mid(
    TYPE_DATA array_in[1073741824],
    TYPE_WEIGHT array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2
);

void tensor_contraction_last(
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

void relu_inplace(
    TYPE_DATA* data,
    int shape
);
