#include "tt_nn.h"

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
){
#pragma HLS ALLOCATION instances=1 function

#pragma HLS ARRAY_MAP variable=input_shape horizontal
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE m_axi depth=1048576 port=bias
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
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

        if (dim_mut == dim - 1) {
            tensor_cont_last(
                mul_array_in,
                weight + dim_mut * weight_offset,
                mul_array_out,
                1, 
                in_shape_0,
                input_shape[dim_mut],
                rank_left * output_shape[dim_mut],
                1
            );
        }
        else{
            tensor_cont_mid(
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
    }
    int output_size = 1;
    for (int i = 0; i < dim; i++) {
        output_size *= output_shape[i];
    }
    for (int i = 0; i < output_size; i++) {
        array_list[array_out_offset + i] += bias[i];
    }
}


void tensor_train_backward(
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
){

#pragma HLS ARRAY_MAP variable=input_shape horizontal
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
    TYPE_DATA* mul_array_in;
    TYPE_DATA* mul_array_out;

    for (int dim_mut = 0; dim_mut < dim; dim_mut++) {
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
            mul_array_out = array_list + grad_in_offset;;
            rank_right = 1;
        }
        else {
            mul_array_out = array_list + tmp_offset + dim_mut * tmp_distance;
            rank_right = rank[dim_mut];
        }
        if (dim_mut == 0) {
            mul_array_in = array_list + grad_out_offset;
            rank_left = 1;
        }
        else {
            mul_array_in = array_list + tmp_offset + (dim_mut-1) * tmp_distance;
            rank_left = rank[dim_mut - 1];
        }

        if (dim_mut == dim - 1) {
            tensor_cont_last(
                mul_array_in,
                weight + dim_mut * weight_offset,
                mul_array_out,
                1, 
                in_shape_0,
                rank_left * output_shape[dim_mut],
                1,
                input_shape[dim_mut] * rank_right
            );
        }
        else{
            tensor_cont_mid(
                mul_array_in,
                weight + dim_mut * weight_offset,
                mul_array_out,
                in_shape_0,
                rank_left * input_shape[dim_mut],
                in_shape_2,
                1,
                output_shape[dim_mut] * rank_right
            );
        }
    }
}

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
){

#pragma HLS ARRAY_MAP variable=input_shape horizontal
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE depth=1048576 port=weight
    TYPE_DATA* mul_array_in;
    TYPE_DATA* mul_array_out;

    if (dim_grad == 0) {
        for (int dim_mut = dim - 1; dim_mut > 0; dim_mut--) {
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
            
            mul_array_out = array_list + tmp_offset + (dim_mut - 1) * tmp_distance;
            rank_left = rank[dim_mut - 1];

            if (dim_mut == dim - 1) {
                tensor_cont_last(
                    mul_array_in,
                    weight + dim_mut * weight_offset,
                    mul_array_out,
                    1, 
                    in_shape_0,
                    input_shape[dim_mut],
                    rank_left * output_shape[dim_mut],
                    1
                );
            }
            else{
                tensor_cont_mid(
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
        }
        int in_shape_3 = 1;
        for (int i = 1; i < dim; i++) {
            in_shape_3 *= output_shape[i];
        }
        tensor_cont_end_backward(
            array_list + tmp_offset,
            array_list + grad_out_offset,
            weight_grad,
            1, 
            1, 
            input_shape[0] * rank[0],
            in_shape_3,
            output_shape[0]
        );
    }
        

    else{
        for (int dim_mut = dim_grad + 1; dim_mut < dim; dim_mut++) {
            int in_shape_0 = 1;
            int in_shape_2 = 1;
            int rank_left;
            int rank_right;
            int w_shape_0;
            for (int i = 0; i < dim_grad; i++) {
                in_shape_0 *= input_shape[i];
            }
            in_shape_0 *= rank[dim_grad - 1] * output_shape[dim_grad];
            if (dim_mut > dim_grad + 1) {
                in_shape_0 *= rank[dim_grad];
            }
            for (int i = dim_grad + 1; i < dim_mut; i++) {
                in_shape_0 *= input_shape[i];
            }
            for (int i = dim_mut + 1; i < dim; i++) {
                in_shape_2 *= output_shape[i];
            }
            mul_array_in = array_list + tmp_offset + (dim_mut-1) * tmp_distance;
            mul_array_out = array_list + tmp_offset + dim_mut * tmp_distance;
            if (dim_mut == dim - 1) {
                rank_right = 1;
            }
            else {
                rank_right = rank[dim_mut];
            }
            
            if (dim_mut == dim_grad + 1) {
                rank_left = 1;
                w_shape_0 = rank[dim_mut - 1];
            }
            else {
                rank_left = rank[dim_mut - 1];
                w_shape_0 = 1;
            }

            if (dim_mut == dim - 1) {
                tensor_cont_last(
                    mul_array_in,
                    weight + dim_mut * weight_offset,
                    mul_array_out,
                    1,
                    in_shape_0,
                    rank_left * input_shape[dim_mut],
                    w_shape_0,
                    output_shape[dim_mut] * rank_right
                );
            
            }
            else{
                tensor_cont_mid(
                    mul_array_in,
                    weight + dim_mut * weight_offset,
                    mul_array_out,
                    in_shape_0,
                    rank_left * input_shape[dim_mut],
                    in_shape_2,
                    w_shape_0,
                    output_shape[dim_mut] * rank_right
                );
            }
        }
    }
    int in_shape_0 = 1;
    int in_shape_3 = 1;
    for (int i = 0; i < dim_grad; i++) {
        in_shape_0 *= input_shape[i];
    }
    for (int i = dim_grad + 1; i < dim; i++) {
        in_shape_3 *= input_shape[i];
    }
    tensor_cont_end_backward(
        array_list + tmp_offset + (dim - 1) * tmp_distance,
        array_list + array_in_offset,
        weight_grad + dim_grad * weight_offset,
        in_shape_0,
        rank[dim_grad - 1] * output_shape[dim_grad],
        rank[dim_grad],
        in_shape_3,
        input_shape[dim_grad]
    );
}

void tensor_train_forward_wrapper(
	TYPE_DATA array_list[1073741824],
	TYPE_WEIGHT weight[1048576],
	TYPE_DATA bias[1048576]
){
	int input_shape[] = {7, 4, 7, 4};
	int hidden_shape0[] = {4, 8, 4, 4};
	int rank0[] = {16, 16, 16};
	int hidden_shape1[] = {32, 16};
	tensor_train_forward(
		array_list,
		weight,
		bias,
		0,
		32*32,
		32*32 + 512,
		input_shape,
		hidden_shape0,
		rank0,
		4,
		8*8*20*20,
		20*32*32
	);
}

//inline int sub2ind3(
//    int ind0,
//    int ind1,
//    int ind2,
//    int size1,
//    int size2
//){
//#pragma HLS INLINE
//    return (ind0 * size1 + ind1) * size2 + ind2;
//}

#define sub2ind3(ind0, ind1, ind2, size1, size2) ((ind0 * size1 + ind1) * size2 + ind2)

inline int sub2ind4(
    int ind0,
    int ind1,
    int ind2,
    int ind3,
    int size1,
    int size2,
    int size3
){
#pragma HLS INLINE
    return ((ind0 * size1 + ind1) * size2 + ind2) * size3 + ind3;
}

void tensor_cont_last_wrapper(
    TYPE_DATA array_in[1073741824],
    TYPE_WEIGHT array_weight[1048576],
    TYPE_DATA array_out[1073741824]
){
#pragma HLS ARRAY_PARTITION variable=array_in cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=array_weight cyclic factor=256
#pragma HLS ARRAY_PARTITION variable=array_out cyclic factor=16

//	int array_in_size_0 = 2;
//	int array_in_size_1 = 2;
//	int array_in_size_2 = 32;
//	int array_weight_size_0 = 4;
//	int array_weight_size_2 = 4;
#define array_in_size_0 2
#define array_in_size_1 2
#define array_in_size_2 32
#define array_weight_size_0 4
#define array_weight_size_2 1

	for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
		for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
			for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
				for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2++) {
					TYPE_INTER res = 0;
					for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2 += PARALLEL_DEGREE) {
#pragma HLS pipeline
						for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
#pragma HLS UNROLL
							int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2+i_in_o,
								array_in_size_1, array_in_size_2);
							int ind_w = sub2ind3(i_w_0, i_in_2+i_in_o, i_w_2, array_in_size_2, array_weight_size_2);
							res += array_in[ind_in] * array_weight[ind_w];
						}
					}
					int ind_out = sub2ind4(i_in_0, i_in_1, i_w_0, i_w_2, array_in_size_1, array_weight_size_0, array_weight_size_2);
					array_out[ind_out] = res;
				}
			}
		}
	}
}//	tensor_cont_last(
//		array_in,
//		array_weight,
//		array_out,
//		32,
//		16,
//		32,
//		16,
//		2
//	);

void tensor_cont_mid_wrapper(
	TYPE_DATA array_in[1073741824],
	TYPE_WEIGHT array_weight[1048576],
	TYPE_DATA array_out[1073741824]
) {
#pragma HLS ARRAY_PARTITION variable=array_in cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=array_weight cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=array_out cyclic factor=16
	tensor_cont_mid(
		array_in,
		array_weight,
		array_out,
		8,
		8,
		32,
		4,
		4
	);
}
