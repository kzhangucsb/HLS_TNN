#include <ap_fixed.h>
#define TYPE_WEIGHT ap_fixed<8, 0>
#define TYPE_DATA ap_fixed<8, 0>

void tensor_contraction_raw(
    TYPE_DATA array_in[78400],
    TYPE_WEIGHT array_weight[10000],
    TYPE_DATA array_out[78400],
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2
);

void tensor_train_forward(
    TYPE_DATA array_list[4800000],
    TYPE_WEIGHT weight[102400],
    TYPE_DATA bias[102400],
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
