#include "tt_nn.h"

inline int sub2ind3(
    int ind0,
    int ind1,
    int ind2,
    int size1,
    int size2
){
    return (ind0 * size1 + ind1) * size2 + ind2;
}

inline int sub2ind4(
    int ind0,
    int ind1,
    int ind2,
    int ind3,
    int size1,
    int size2,
    int size3
){
    return ((ind0 * size1 + ind1) * size2 + ind2) * size3 + ind3;
}

void tensor_contraction_raw(
    TYPE_DATA* array_in,
    TYPE_WEIGHT* array_weight,
    TYPE_DATA* array_out,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2
){
    /* tensor contraction on the second dimension
    * array_in: size (array_in_size_0*array_in_size_1*array_in_size_2),
    * array_weight: size (array_weight_size_0*array_in_size_1*array_weight_size_2),
    * array_out: size(array_in_size_0*array_weight_size_0*array_weight_size_2*array_in_size_2)
    * All arrays are in C order
    */
    TYPE_DATA res;
    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
        for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
            for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2++) {
                for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2++) {
                    res = 0;
                    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1 += 1) {
                        int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2, array_in_size_1, array_in_size_2);
                        int ind_w = sub2ind3(i_w_0, i_in_1, i_w_2, array_in_size_1, array_weight_size_2);
                        res += array_in[ind_in] * array_weight[ind_w];
                    }
                    int ind_out = sub2ind4(i_in_0, i_w_0, i_w_2,  i_in_2,
                        array_weight_size_0, array_weight_size_2, array_in_size_2);
                    array_out[ind_out] = res;
                }
            }
        }
    }
}

void tensor_train_forward(
    TYPE_DATA* array_in,
    TYPE_WEIGHT** weight,
    TYPE_DATA* bias,
    TYPE_DATA* array_out,
    TYPE_DATA** tmp,
    int* input_shape,
    int* output_shape,
    int* rank,
    int dim
){
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
            mul_array_in = array_in;
            rank_right = 1;
        }
        else {
            mul_array_in = tmp[dim_mut];
            rank_right = rank[dim_mut];
        }
        if (dim_mut == 0) {
            mul_array_out = array_out;
            rank_left = 1;
        }
        else {
            mul_array_out = tmp[dim_mut - 1];
            rank_left = rank[dim_mut - 1];
        }
        tensor_contraction_raw(
            mul_array_in,
            weight[dim_mut],
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
        array_out[i] += bias[i];
    }
}



