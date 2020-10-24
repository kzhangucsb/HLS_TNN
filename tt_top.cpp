#include "tt_nn.h"

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
){

#pragma HLS ARRAY_MAP variable=input_shape horizontal
#pragma HLS INTERFACE m_axi depth=4800000 port=array_list offset=slave
#pragma HLS INTERFACE m_axi depth=102400 port=bias
#pragma HLS INTERFACE m_axi depth=102400 port=weight
#pragma HLS INTERFACE m_axi depth=87400 port=array_in offset=slave
    TYPE_DATA* mul_array_in;
    TYPE_DATA* mul_array_out;

    for (int dim_mut = dim - 1; dim_mut >= 0; dim_mut--) {
        int in_shape_0 = 1;
        int in_shape_2 = 1;
        int rank_left;
        int rank_right;
        for (int i = 0; i < dim_mut; i++) {
            in_shape_0 *= input_shape[i];
        }
        for (int i = dim_mut + 1; i < dim; i++) {
            in_shape_2 *= output_shape[i];
        }

        if (dim_mut == dim - 1) {
            mul_array_in = array_list + array_in_offset;
            rank_right = 1;
        }
        else {
            mul_array_in = array_list + tmp_offset + dim_mut * tmp_distance;
            rank_right = rank[dim_mut];
        }
        if (dim_mut == 0) {
            mul_array_out = array_list + array_out_offset;;
            rank_left = 1;
        }
        else {
            mul_array_out = array_list + tmp_offset + (dim_mut - 1) * tmp_distance;
            rank_left = rank[dim_mut - 1];
        }
        tensor_contraction_raw(
            mul_array_in,
            weight + dim_mut * weight_offset,
            mul_array_out,
            in_shape_0,
            input_shape[dim_mut] * rank_right,
            in_shape_2,
            rank_left * output_shape[dim_mut],
            1
        );
    }
    int output_size = 1;
    for (int i = 0; i < dim; i++) {
        output_size *= output_shape[i];
    }
    for (int i = 0; i < output_size; i++) {
        array_list[array_out_offset + i] += bias[i];
    }
}



