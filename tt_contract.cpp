#include "tt_nn.h"
#ifndef SYNTHESIS
#include <assert.h>
#endif

inline int sub2ind3(
    int ind0,
    int ind1,
    int ind2,
    int size1,
    int size2
){
#pragma HLS INLINE
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
#pragma HLS INLINE
    return ((ind0 * size1 + ind1) * size2 + ind2) * size3 + ind3;
}

void tensor_cont_mid(
    TYPE_DATA array_in[1073741824],
    TYPE_WEIGHT array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2
){
    /* tensor contraction on the second dimension
    ABCxDBE->ADEC
    * array_in: size (array_in_size_0*array_in_size_1*array_in_size_2),
    * array_weight: size (array_weight_size_0*array_in_size_1*array_weight_size_2),
    * array_out: size(array_in_size_0*array_weight_size_0*array_weight_size_2*array_in_size_2)
    * All arrays are in C order
    */
    #ifndef SYNTHESIS
    assert (array_in_size_2 % PARALLEL_DEGREE == 0);
    #endif 
    TYPE_INTER res[PARALLEL_DEGREE];
    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
        for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
            for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2+=PARALLEL_DEGREE) {
				for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2++) {
#pragma HLS dataflow
					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
                        res[i_in_o] = 0;
					}
					for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1 += 1) {
#pragma HLS pipeline
						int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2, array_in_size_1, array_in_size_2);
						int ind_w = sub2ind3(i_w_0, i_in_1, i_w_2, array_in_size_1, array_weight_size_2);
						for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
							res[i_in_o] += array_in[ind_in + i_in_o] * array_weight[ind_w];
						}
					}
					int ind_out = sub2ind4(i_in_0, i_w_0, i_w_2,  i_in_2,
						array_weight_size_0, array_weight_size_2, array_in_size_2);
					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
                        array_out[ind_out+i_in_o] = res[i_in_o];
                    }
                }
            }
        }
    }
}

void tensor_cont_last(
    TYPE_DATA array_in[1073741824],
    TYPE_WEIGHT array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_1
){
    /* tensor contraction on the first and last dimension
    ABCxBDC->AD
    */
    #ifndef SYNTHESIS
    assert (array_in_size_2 % PARALLEL_DEGREE == 0);
    #endif 
    //TYPE_INTER res;
    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
        for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1++) {
            TYPE_INTER res = 0;
            for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
                for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2 += PARALLEL_DEGREE) {
#pragma HLS PIPELINE
                    int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2,  array_in_size_1, array_in_size_2);
                    int ind_w = sub2ind3(i_in_1, i_w_1, i_in_2, array_weight_size_1, array_in_size_2);
                    for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
#pragma HLS UNROLL
                        res += array_in[ind_in+i_in_o] * array_weight[ind_w+i_in_o];
                    }
                }
            }
            int ind_out = sub2ind3(0, i_in_0, i_w_1, array_in_size_0, array_weight_size_1);
                array_out[ind_out] = res;
        }
    }
}


void tensor_cont_end_backward(
    TYPE_DATA array_in[1073741824],
    TYPE_DATA array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_in_size_3,
    int array_weight_size_1
){
    /* tensor contraction on the first and last dimension
    ABCDxAED->BEC
    */
    #ifndef SYNTHESIS
    assert (array_in_size_3 % PARALLEL_DEGREE == 0);
    #endif 
    TYPE_INTER res;
    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
        for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2++) {
            for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1++) {
                res = 0;
                for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
                    for (int i_in_3 = 0; i_in_3 < array_in_size_3; i_in_3 += PARALLEL_DEGREE) {
                        int ind_in = sub2ind4(i_in_0, i_in_1, i_in_2, i_in_3, 
                            array_in_size_1, array_in_size_2, array_in_size_3);
                        int ind_w = sub2ind3(i_in_0, i_w_1, i_in_3, array_weight_size_1, array_in_size_3);
                        for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
#pragma HLS UNROLL
                            res += array_in[ind_in+i_in_o] * array_weight[ind_w+i_in_o];
                        }
                    }
                }
                int ind_out = sub2ind3(i_in_1, i_w_1, i_in_2, array_weight_size_1, array_in_size_2);
                assert (ind_out < 8*8*20*20);
                array_out[ind_out] += res;
            }
        }
    }
}

void tensor_cont_head_backward(
    TYPE_DATA array_in[1073741824],
    TYPE_DATA array_weight[1048576],
    TYPE_DATA array_out[1073741824],
    int array_in_size_0,
    int array_in_size_1,
    int array_weight_size_1
){
    /* tensor contraction on the first and last dimension
    ABxAE->BE
    */
    #ifndef SYNTHESIS
    assert (array_in_size_1 % PARALLEL_DEGREE == 0);
    #endif 
    TYPE_INTER res[PARALLEL_DEGREE];
    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
        for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1+= PARALLEL_DEGREE) {
            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
                res[i_in_o] = 0;
            }
            for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
                int ind_in = sub2ind3(0, i_in_0, i_in_1, array_in_size_0, array_in_size_1);
                int ind_w = sub2ind3(0, i_in_0, i_w_1, array_in_size_0, array_weight_size_1);
                for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
                    res[i_in_o] += array_in[ind_in] * array_weight[ind_w + i_in_o];
                }
            }
            int ind_out = sub2ind3(0, i_in_1, i_w_1, array_in_size_1, array_weight_size_1);
            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
                array_out[ind_out + i_in_o] += res[i_in_o];
            }
        }
    }
}
