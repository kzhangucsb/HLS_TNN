#include "tt_nn.h"
#include <assert.h>
// #include <iostream>
// #include <string.h>

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
){
#pragma HLS ALLOCATION instances=1 function

#pragma HLS ARRAY_MAP variable=input_shape horizontal
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE m_axi depth=1048576 port=bias
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
    // TYPE_DATA* mul_array_in;
    // TYPE_DATA* mul_array_out;

    for (int dim_mut = dim - 1; dim_mut >= 0; dim_mut--) {
        int in_shape_0 = 1;
        int in_shape_2 = 1;
        int rank_left;
        int rank_right;
        // char in_base[100];
        // char out_base[100];
        int in_offset;
        int out_offset;
        for (int i = 0; i < dim_mut; i++) {
            in_shape_0 *= input_shape[i];
        }
        for (int i = dim_mut + 1; i < dim; i++) {
            in_shape_2 *= output_shape[i];
        }

        if (dim_mut == dim - 1) {
            //mul_array_in = array_list + array_in_offset;
            // strcpy(in_base, "array_list");
            in_offset = array_in_offset;
            rank_right = 1;
        }
        else {
            // mul_array_in = array_list + tmp_offset + dim_mut * tmp_distance;
            // strcpy(in_base, "array_list");
            in_offset = tmp_offset + dim_mut * tmp_distance;
            rank_right = rank[dim_mut];
        }
        if (dim_mut == 0) {
            // mul_array_out = array_list + array_out_offset;
            // strcpy(out_base, "array_list");
            out_offset = array_out_offset;
            rank_left = 1;
        }
        else {
            // mul_array_out = array_list + tmp_offset + (dim_mut - 1) * tmp_distance;
            // strcpy(out_base, "array_list");
            out_offset = tmp_offset + (dim_mut - 1) * tmp_distance;
            rank_left = rank[dim_mut - 1];
        }

        if (dim_mut == dim - 1) {
            // std::cout << "tensor_cont_last(" << in_base << "+" << in_offset;
            // std::cout << ", " << out_base << "+" << out_offset << " ";
            tensor_cont_last(
                array_list,
                weight,
                in_offset,
                out_offset,
                weight_offset + dim_mut * weight_distance,
                in_shape_0,
                1,
                input_shape[dim_mut],
                rank_left * output_shape[dim_mut]
            );
        }
        else{
            // std::cout << "tensor_cont_mid(" << in_base << "+" << in_offset;
            // std::cout << ", " << out_base << "+" << out_offset << " ";
            tensor_cont_mid(
                array_list,
                weight,
                in_offset,
                out_offset,
                weight_offset + dim_mut * weight_distance,
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
        array_list[array_out_offset + i] += bias[bias_offset + i];
    }
}


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
){

#pragma HLS ARRAY_MAP variable=input_shape horizontal
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
    // TYPE_DATA* mul_array_in;
    // TYPE_DATA* mul_array_out;
    // char in_base[100];
    // char out_base[100];
    int in_offset;
    int out_offset;

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
            // mul_array_out = array_list + grad_in_offset;
            // strcpy(out_base, "array_list");
            out_offset = grad_in_offset;
            rank_right = 1;
        }
        else {
            // mul_array_out = array_list + tmp_offset + dim_mut * tmp_distance;
            // strcpy(out_base, "array_list");
            out_offset = tmp_offset + dim_mut * tmp_distance;
            rank_right = rank[dim_mut];
        }
        if (dim_mut == 0) {
            // mul_array_in = array_list + grad_out_offset;
            // strcpy(in_base, "array_list");
            in_offset = grad_out_offset;
            rank_left = 1;
        }
        else {
            // mul_array_in = array_list + tmp_offset + (dim_mut-1) * tmp_distance;
            // strcpy(in_base, "array_list");
            in_offset = tmp_offset + (dim_mut-1) * tmp_distance;
            rank_left = rank[dim_mut - 1];
        }

        if (dim_mut == dim - 1) {
            // std::cout << "tensor_cont_last(" << in_base << "+" << in_offset;
            // std::cout << ", " << out_base << "+" << out_offset << " ";
            tensor_cont_last(
                array_list,
                weight,
                in_offset,
                out_offset,
                weight_offset + dim * weight_distance,
                in_shape_0,
                rank_left,
                output_shape[dim_mut],
                input_shape[dim_mut]
            );
        }
        else{
            // std::cout << "tensor_cont_mid(" << in_base << "+" << in_offset;
            // std::cout << ", " << out_base << "+" << out_offset << " ";
            tensor_cont_mid(
                array_list,
                weight,
                in_offset,
                out_offset,
                weight_offset + dim_mut * weight_distance,
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
){

#pragma HLS ARRAY_MAP variable=input_shape horizontal
#pragma HLS INTERFACE m_axi depth=1073741824 port=array_list offset=slave
#pragma HLS INTERFACE depth=1048576 port=weight
    // TYPE_DATA* mul_array_in;
    // TYPE_DATA* mul_array_out;
    // char in_base[100];
    // char out_base[100];
    int in_offset;
    int out_offset;


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

            // mul_array_out = array_list + tmp_offset + (dim_mut - 1) * tmp_distance;
            // strcpy(out_base, "array_list");
            out_offset = tmp_offset + (dim_mut - 1) * tmp_distance;
            if (dim_mut == dim - 1) {  
                // std::cout << "tensor_cont_last(" << "array_list" << "+" << array_in_offset;
                // std::cout << ", " << out_base << "+" << out_offset << " ";  
                tensor_cont_last(
                    array_list,
                    weight,
                    array_in_offset,
                    out_offset,
                    weight_offset + dim_mut * weight_distance,
                    in_shape_0,
                    1,
                    input_shape[dim_mut],
                    rank[dim_mut - 1] * output_shape[dim_mut]
                );
            } // dim_mut == dim - 1
            else{
                //mul_array_in = array_list + tmp_offset + dim_mut * tmp_distance;
                //rank_right = rank[dim_mut];
                // std::cout << "tensor_cont_last(" << "array_list" << "+" << tmp_offset + dim_mut * tmp_distance,;
                // std::cout << ", " << out_base << "+" << out_offset << " ";  
                tensor_cont_mid(
                    array_list,
                    weight,
                    tmp_offset + dim_mut * tmp_distance,
                    out_offset,
                    weight_offset + dim_mut * weight_distance,
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
        // std::cout << "tensor_cont_end_backward(" << "array_list" << "+" << tmp_offset;
        // std::cout << ", " << "weight_grad ";  
        tensor_cont_end_backward(
            array_list,
            weight_grad,
            tmp_offset,
            grad_out_offset,
            weight_offset,
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
            // mul_array_in = array_list + tmp_offset + (dim_mut - 2) * tmp_distance;
            // mul_array_out = array_list + tmp_offset + (dim_mut - 1) * tmp_distance;
            // strcpy(in_base, "array_list");
            in_offset = tmp_offset + (dim_mut - 2) * tmp_distance;
            // strcpy(out_base, "array_list");
            out_offset = tmp_offset + (dim_mut - 1) * tmp_distance;
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
                // std::cout << "tensor_cont_last(" << in_base << "+" << in_offset;
                // std::cout << ", " << out_base << "+" << out_offset << " ";  
                tensor_cont_last(
                    array_list,
                    weight,
                    in_offset,
                    out_offset,
                    weight_offset + dim * weight_distance,
                    in_shape_0,
                    rank_left,
                    output_shape[dim_mut],
                    w_shape_0 * input_shape[dim_mut]
                );
            
            }
            else{
                // std::cout << "tensor_cont_mid(" << in_base << "+" << in_offset;
                // std::cout << ", " << out_base << "+" << out_offset << " ";  
                tensor_cont_mid(
                    array_list,
                    weight,
                    in_offset,
                    out_offset,
                    weight_offset + dim_mut * weight_distance,
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
                array_list,
                weight_grad,
                tmp_offset + (dim - 2) * tmp_distance,
                array_in_offset,
                weight_offset + dim_grad * weight_distance,
                in_shape_0,
                rank[dim_grad - 1] * output_shape[dim_grad],
                input_shape[dim_grad]
            );

        }
        else{
            tensor_cont_end_backward(
                array_list,
                weight_grad,
                tmp_offset + (dim - 2) * tmp_distance,
                array_in_offset,
                weight_offset + dim_grad * weight_distance,
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
){
    tensor_train_input_grad(
        array_list,
        weight,
        input_shape,
        output_shape,
        rank,
        dim,
        grad_out_offset,
        grad_in_offset,
        tmp_offset,
        weight_offset,
        tmp_distance,
        weight_distance
    );
    for (int dim_grad = dim - 1; dim_grad >= 0; dim_grad--) {
        tensor_train_weight_grad(
            array_list,
            weight,
            weight_grad,
            input_shape,
            output_shape,
            rank,
            dim,
            dim_grad,
            array_in_offset,
            grad_out_offset,
	        tmp_offset,
	        weight_offset,
	        tmp_distance,
            weight_distance
        );
    }
}
