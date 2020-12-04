#include "tt_nn.h"
#include <assert.h>

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
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=bias
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
#pragma HLS ARRAY_RESHAPE variable=array_list cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=weight cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=bias cyclic factor=16 dim=1
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
                in_shape_0,
                1,
                input_shape[dim_mut],
                rank_left * output_shape[dim_mut]
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
){
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
#pragma HLS ARRAY_RESHAPE variable=array_list cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=weight cyclic factor=16 dim=1
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
                weight + dim * weight_offset,
                mul_array_out,
                in_shape_0,
                rank_left,
                output_shape[dim_mut],
                input_shape[dim_mut]
            );
        }
        else{
            tensor_cont_mid(
                mul_array_in,
                weight + dim_mut * weight_offset,
                mul_array_out,
                in_shape_0,
                rank_left * output_shape[dim_mut],
                in_shape_2,
                1,
                input_shape[dim_mut] * rank_right
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
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight_grad
#pragma HLS ARRAY_RESHAPE variable=array_list cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=weight cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=weight_grad cyclic factor=16 dim=1
    TYPE_DATA* mul_array_in;
    TYPE_DATA* mul_array_out;

    if (dim_grad == 0) {
        for (int dim_mut = dim - 1; dim_mut > 0; dim_mut--) {
            int in_shape_0 = 1;
            int in_shape_2 = 1;
            //int rank_left;
            //int rank_right;
            for (int i = 0; i < dim_mut; i++) {
                in_shape_0 *= input_shape[i];
            }
            for (int i = dim_mut + 1; i < dim; i++) {
                in_shape_2 *= output_shape[i];
            }
            
            //rank_left = rank[dim_mut - 1];

            mul_array_out = array_list + tmp_offset + (dim_mut - 1) * tmp_distance;
            if (dim_mut == dim - 1) {    
                tensor_cont_last(
                    array_list + array_in_offset,
                    weight + dim_mut * weight_offset,
                    mul_array_out,
                    in_shape_0,
                    1,
                    input_shape[dim_mut],
                    rank[dim_mut - 1] * output_shape[dim_mut]
                );
            } // dim_mut == dim - 1
            else{
                //mul_array_in = array_list + tmp_offset + dim_mut * tmp_distance;
                //rank_right = rank[dim_mut];
                tensor_cont_mid(
                    array_list + tmp_offset + dim_mut * tmp_distance,
                    weight + dim_mut * weight_offset,
                    mul_array_out,
                    in_shape_0,
                    input_shape[dim_mut] * rank[dim_mut],
                    in_shape_2,
                    rank[dim_mut - 1] * output_shape[dim_mut],
                    1
                );
            } // dim_mul < dim - 1
        } // for (int dim_mut = dim - 1; dim_mut > 0; dim_mut--)
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
    } // dim_grad == 0
        
    // dim_grad > 0
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
            mul_array_in = array_list + tmp_offset + (dim_mut - 2) * tmp_distance;
            mul_array_out = array_list + tmp_offset + (dim_mut - 1) * tmp_distance;
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
                    weight + dim * weight_offset,
                    mul_array_out,
                    in_shape_0,
                    rank_left,
                    output_shape[dim_mut],
                    w_shape_0 * input_shape[dim_mut]
                );
            
            }
            else{
                tensor_cont_mid(
                    mul_array_in,
                    weight + dim_mut * weight_offset,
                    mul_array_out,
                    in_shape_0,
                    rank_left * output_shape[dim_mut],
                    in_shape_2,
                    w_shape_0,
                    input_shape[dim_mut] * rank_right
                );
            }
        } // for (int dim_mut = dim_grad + 1; dim_mut < dim; dim_mut++)
        int in_shape_0 = 1;
        int in_shape_3 = 1;
        for (int i = 0; i < dim_grad; i++) {
            in_shape_0 *= input_shape[i];
        }
        for (int i = dim_grad + 1; i < dim; i++) {
            in_shape_3 *= input_shape[i];
        }
        int rank_right;
        if (dim_grad == dim - 1){
            rank_right = 1;
        }
        else {
            rank_right = rank[dim_grad];
        }
        if (dim_grad == dim - 1) {
            tensor_cont_head_backward(
                array_list + tmp_offset + (dim - 2) * tmp_distance,
                array_list + array_in_offset,
                weight_grad + dim_grad * weight_offset,
                in_shape_0,
                rank[dim_grad - 1] * output_shape[dim_grad],
                input_shape[dim_grad]
            );

        }
        else{
            tensor_cont_end_backward(
                array_list + tmp_offset + (dim - 2) * tmp_distance,
                array_list + array_in_offset,
                weight_grad + dim_grad * weight_offset,
                in_shape_0,
                rank[dim_grad - 1] * output_shape[dim_grad],
                rank_right,
                in_shape_3,
                input_shape[dim_grad]
            );
        }
    }// dim_grad > 0
    
}

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
){
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight_grad
#pragma HLS ARRAY_RESHAPE variable=array_list cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=weight cyclic factor=16 dim=1
#pragma HLS ARRAY_RESHAPE variable=weight_grad cyclic factor=16 dim=1
    tensor_train_input_grad(
        array_list,
        weight,
        grad_out_offset,
        grad_in_offset,
        tmp_offset,
        input_shape,
        output_shape,
        rank,
        dim,
        weight_offset,
        tmp_distance
    );
    for (int dim_grad = dim - 1; dim_grad >= 0; dim_grad--) {
        tensor_train_weight_grad(
            array_list,
            weight,
            weight_grad,
            dim_grad,
            array_in_offset,
            grad_out_offset,
	        tmp_offset,
            input_shape,
            output_shape,
            rank,
            dim,
	        weight_offset,
	        tmp_distance
        );
    }
}












