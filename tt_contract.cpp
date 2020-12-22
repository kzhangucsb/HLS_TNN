#include "tt_nn.h"
#ifndef SYNTHESIS
#include <assert.h>
#include <iostream>
using namespace std;
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

void tensor_cont_mid_compute(
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
    int shift,
    TYPE_DATA local[1024][PARALLEL_DEGREE],
    int i_in_0,
    int i_in_2
){
    TYPE_INTER res[PARALLEL_DEGREE];
#pragma HLS ARRAY_RESHAPE variable=res complete
#pragma HLS ARRAY_RESHAPE variable=local dim=2
    for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
        for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2++) {
            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
                res[i_in_o] = 0;
            }
            loop_in_1: for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1 += 1) {
#pragma HLS pipeline
                int ind_w = sub2ind3(i_w_0, i_in_1, i_w_2, array_in_size_1, array_weight_size_2);
                for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
//							res[i_in_o] += array[(in_offset + ind_in)/ PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
//												 * weight[weight_offset + ind_w];
                    res[i_in_o] += local[i_in_1][i_in_o] * weight[weight_offset + ind_w];
                }
            }
            int ind_out = sub2ind4(i_in_0, i_w_0, i_w_2,  i_in_2,
                array_weight_size_0, array_weight_size_2, array_in_size_2);
            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#ifdef SYNTHESIS
                array[(out_offset + ind_out) / PARALLEL_DEGREE * PARALLEL_DEGREE +i_in_o] = res[i_in_o] >> shift;
#else
                array[(out_offset + ind_out)  +i_in_o] = res[i_in_o] / pow(2, shift);
#endif
            }
        }
    }
}

void tensor_cont_mid_load(
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
	int shift,
    TYPE_DATA local[1024][PARALLEL_DEGREE], 
    int i_in_0,
    int i_in_2
){
    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
        int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2, array_in_size_1, array_in_size_2);
        for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
            local[i_in_1][i_in_o] = array[(in_offset + ind_in)/ PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
        }
    }
}

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
){
    /* tensor contraction on the second dimension
    ABCxDBE->ADEC
    * array_in: size (array_in_size_0*array_in_size_1*array_in_size_2),
    * array_weight: size (array_weight_size_0*array_in_size_1*array_weight_size_2),
    * array_out: size(array_in_size_0*array_weight_size_0*array_weight_size_2*array_in_size_2)
    * All arrays are in C order
    */
//#pragma HLS stable variable=array_in
//#pragma HLS stable variable=array_out
#pragma HLS INTERFACE m_axi depth=1073741824 port=array offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
#pragma HLS ARRAY_RESHAPE variable=array cyclic factor=16
#pragma HLS ARRAY_RESHAPE variable=weight cyclic factor=16
#pragma HLS DEPENDENCE array inter false
    #ifndef SYNTHESIS
    assert (array_in_size_2 % PARALLEL_DEGREE == 0);
    #endif 
    //TYPE_INTER res[PARALLEL_DEGREE];
    TYPE_DATA locall[1024][PARALLEL_DEGREE];
    TYPE_DATA localr[1024][PARALLEL_DEGREE];
#pragma HLS ARRAY_RESHAPE variable=locall dim=2
#pragma HLS ARRAY_RESHAPE variable=localr dim=2
#pragma HLS resource variable=locall core=RAM_1P
#pragma HLS resource variable=localr core=RAM_1P
#pragma HLS ARRAY_RESHAPE variable=locall dim=2
#pragma HLS ARRAY_RESHAPE variable=localr dim=2
for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2+=PARALLEL_DEGREE) {
	tensor_cont_mid_load(
		array, weight, in_offset, out_offset, weight_offset,
		array_in_size_0, array_in_size_1, array_in_size_2,
		array_weight_size_0, array_weight_size_2, shift,
		localr, 0, i_in_2
	);
	for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
		if (i_in_0 % 2 == 0){
			tensor_cont_mid_load(
				array, weight, in_offset, out_offset, weight_offset,
				array_in_size_0, array_in_size_1, array_in_size_2,
				array_weight_size_0, array_weight_size_2, shift,
				locall, i_in_0 + 1, i_in_2
			);
			tensor_cont_mid_compute(
				array, weight, in_offset, out_offset, weight_offset,
				array_in_size_0, array_in_size_1, array_in_size_2,
				array_weight_size_0, array_weight_size_2, shift,
				localr, i_in_0, i_in_2
			);
		}
		else {
			tensor_cont_mid_load(
				array, weight, in_offset, out_offset, weight_offset,
				array_in_size_0, array_in_size_1, array_in_size_2,
				array_weight_size_0, array_weight_size_2, shift,
				localr, i_in_0 + 1, i_in_2
			);
			tensor_cont_mid_compute(
				array, weight, in_offset, out_offset, weight_offset,
				array_in_size_0, array_in_size_1, array_in_size_2,
				array_weight_size_0, array_weight_size_2, shift,
				locall, i_in_0, i_in_2
			);
		}
	}
}

void tensor_cont_last_load(
    TYPE_DATA array[1073741824],
    TYPE_WEIGHT weight[1048576],
    int in_offset,
    int out_offset,
    int weight_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_1,
	int shift,
    TYPE_DATA local[32][16],
    int i_in_0
){
    
#pragma HLS ARRAY_PARTITION variable=local dim=2 factor=16
#pragma HLS resource variable=local core=RAM_1P
    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
        for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2 += PARALLEL_DEGREE) {
            int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2,  array_in_size_1, array_in_size_2);
            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
#pragma HLS UNROLL
                local[i_in_1][i_in_2 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
                                = array[(in_offset + ind_in) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
            }
        }
    }
}
void tensor_cont_last_compute(
    TYPE_DATA array[1073741824],
    TYPE_WEIGHT weight[1048576],
    int in_offset,
    int out_offset,
    int weight_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_1,
	int shift,
    TYPE_DATA local[14336][16],
    int i_in_0
){
for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1++) {
        TYPE_INTER res = 0;
        loop_in_1: for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
            loop_in_2: for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2 += PARALLEL_DEGREE) {
#pragma HLS PIPELINE

                int ind_w = sub2ind3(i_in_1, i_w_1, i_in_2, array_weight_size_1, array_in_size_2);
                for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
#pragma HLS UNROLL
//                        res += array[(in_offset + ind_in) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
//									 * weight[(weight_offset+ind_w) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
                    res += local[i_in_1][i_in_2 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
                                * weight[(weight_offset+ind_w)/ PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
                }
            }
        }
        int ind_out = sub2ind3(0, i_in_0, i_w_1, array_in_size_0, array_weight_size_1);
#ifdef SYNTHESIS
        array[out_offset + ind_out] = res >> shift;
#else
        array[out_offset + ind_out] = res / pow(2, shift);
#endif
    }
}

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
){
    /* tensor contraction on the first and last dimension
    ABCxBDC->AD
    */
#pragma HLS INTERFACE m_axi depth=1073741824 port=array offset=slave
#pragma HLS INTERFACE ap_memory depth=1048576 port=weight
#pragma HLS ARRAY_RESHAPE variable=array cyclic factor=16
#pragma HLS ARRAY_RESHAPE variable=weight cyclic factor=16
#pragma HLS DEPENDENCE array inter false
    #ifndef SYNTHESIS
    assert (array_in_size_2 % PARALLEL_DEGREE == 0);
    cout << "tensor_cont_last(array, weight, ";
    #endif 
    //TYPE_INTER res;

    TYPE_DATA locall[14336][16];
    TYPE_DATA localr[14336][16];

#pragma HLS ARRAY_PARTITION variable=local dim=2 factor=16
#pragma HLS resource variable=locall core=RAM_1P
#pragma HLS resource variable=localr core=RAM_1P
    tensor_cont_last_load(
        array,
        weight,
        in_offset,
        out_offset,
        weight_offset,
        array_in_size_0,
        array_in_size_1,
        array_in_size_2,
        array_weight_size_1,
        shift,
        localr,
        0
    );
    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
    	//load
        if (i_in_0 % 2 == 0) {
            tensor_cont_last_load(
                array,
                weight,
                in_offset,
                out_offset,
                weight_offset,
                array_in_size_0,
                array_in_size_1,
                array_in_size_2,
                array_weight_size_1,
                shift,
                locall,
                i_in_0 + 1
            );
            tensor_cont_last_compute(
                array,
                weight,
                in_offset,
                out_offset,
                weight_offset,
                array_in_size_0,
                array_in_size_1,
                array_in_size_2,
                array_weight_size_1,
                shift,
                localr,
                i_in_0
            );
        }
        else {
            tensor_cont_last_load(
                array,
                weight,
                in_offset,
                out_offset,
                weight_offset,
                array_in_size_0,
                array_in_size_1,
                array_in_size_2,
                array_weight_size_1,
                shift,
                localr,
                i_in_0 + 1
            );
            tensor_cont_last_compute(
                array,
                weight,
                in_offset,
                out_offset,
                weight_offset,
                array_in_size_0,
                array_in_size_1,
                array_in_size_2,
                array_weight_size_1,
                shift,
                locall,
                i_in_0
            );
        } 
    }
}

void tensor_cont_end_backward_load(
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
	int shift,
    TYPE_DATA local[32][128],
    int i_in_1,
    int i_in_2
)
{
    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
        for (int i_in_3 = 0; i_in_3 < array_in_size_3; i_in_3 += PARALLEL_DEGREE) {
#pragma HLS PIPELINE
            int ind_in = sub2ind4(i_in_0, i_in_1, i_in_2, i_in_3,
                    array_in_size_1, array_in_size_2, array_in_size_3);
            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
#pragma HLS UNROLL
                local[i_in_0][i_in_3 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
                            = array[(a1_offset+ind_in) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
            }
        }
    }
}
void tensor_cont_end_backward_compute(
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
	int shift,
    TYPE_DATA local[32][128],
    int i_in_1,
    int i_in_2
){
    TYPE_INTER res;
    for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1++) {
        res = 0;
        for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
            for (int i_in_3 = 0; i_in_3 < array_in_size_3; i_in_3 += PARALLEL_DEGREE) {
#pragma HLS PIPELINE
//                        int ind_in = sub2ind4(i_in_0, i_in_1, i_in_2, i_in_3,
//                            array_in_size_1, array_in_size_2, array_in_size_3);
                int ind_w = sub2ind3(i_in_0, i_w_1, i_in_3, array_weight_size_1, array_in_size_3);
                for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
#pragma HLS UNROLL
                    //res += array[(a1_offset+ind_in) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
                    res += local[i_in_0][i_in_3 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
                                    * array[(a2_offset+ind_w) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
                }
            }
        }
        int ind_out = sub2ind3(i_in_1, i_w_1, i_in_2, array_weight_size_1, array_in_size_2);
#ifdef SYNTHESIS
        grad_out[out_offset+ind_out] += res >> shift;
#else
        grad_out[out_offset+ind_out] += res / pow(2, shift);
#endif
    }

}

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
){
    /* tensor contraction on the first and last dimension
    ABCDxAED->BEC
    */
#pragma HLS INTERFACE m_axi depth=1073741824 port=array offset=slave
#pragma HLS INTERFACE m_axi depth=1048576 port=grad_out
#pragma HLS ARRAY_RESHAPE variable=array cyclic factor=16
#pragma HLS ARRAY_RESHAPE variable=grad_out cyclic factor=16
#pragma HLS DEPENDENCE array inter false
    #ifndef SYNTHESIS
    assert (array_in_size_3 % PARALLEL_DEGREE == 0);
    #endif 
    
    TYPE_DATA locall[32][128];
    TYPE_DATA localr[32][128];
#pragma HLS ARRAY_PARTITION variable=local dim=2 factor=16
#pragma HLS resource variable=locall core=RAM_1P
#pragma HLS resource variable=localr core=RAM_1P
    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
        tensor_cont_end_backward_load(
            array,
            grad_out,
            a1_offset,
            a2_offset,
            out_offset,
            array_in_size_0,
            array_in_size_1,
            array_in_size_2,
            array_in_size_3,
            array_weight_size_1,
            shift,
            localr,
            i_in_1,
            0
        );
        for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2++) {
        	if (i_in_2 % 2 == 0) {
                tensor_cont_end_backward_load(
                    array,
                    grad_out,
                    a1_offset,
                    a2_offset,
                    out_offset,
                    array_in_size_0,
                    array_in_size_1,
                    array_in_size_2,
                    array_in_size_3,
                    array_weight_size_1,
                    shift,
                    locall,
                    i_in_1,
                    i_in_2 + 1
                );
                tensor_cont_end_backward_compute(
                    array,
                    grad_out,
                    a1_offset,
                    a2_offset,
                    out_offset,
                    array_in_size_0,
                    array_in_size_1,
                    array_in_size_2,
                    array_in_size_3,
                    array_weight_size_1,
                    shift,
                    localr,
                    i_in_1,
                    i_in_2
                );
            }
            else {
                tensor_cont_end_backward_load(
                    array,
                    grad_out,
                    a1_offset,
                    a2_offset,
                    out_offset,
                    array_in_size_0,
                    array_in_size_1,
                    array_in_size_2,
                    array_in_size_3,
                    array_weight_size_1,
                    shift,
                    localr,
                    i_in_1,
                    i_in_2 + 1
                );
                tensor_cont_end_backward_compute(
                    array,
                    grad_out,
                    a1_offset,
                    a2_offset,
                    out_offset,
                    array_in_size_0,
                    array_in_size_1,
                    array_in_size_2,
                    array_in_size_3,
                    array_weight_size_1,
                    shift,
                    locall,
                    i_in_1,
                    i_in_2
                );
            }
        }
    }
}

void tensor_cont_head_backward_load(
    TYPE_DATA array[1073741824],
    TYPE_GRAD grad_out[1073741824],
    int a1_offset,
    int a2_offset,
    int out_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_weight_size_1,
	int shift,
    TYPE_DATA local[256],
    int i_in_1
){
    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
#pragma HLS PIPELINE
        int ind_in = sub2ind3(0, i_in_0, i_in_1, array_in_size_0, array_in_size_1);
        local[i_in_0] = array[a1_offset+ind_in];
    }
}

void tensor_cont_head_backward_compute(
    TYPE_DATA array[1073741824],
    TYPE_GRAD grad_out[1073741824],
    int a1_offset,
    int a2_offset,
    int out_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_weight_size_1,
	int shift,
    TYPE_DATA local[256],
    int i_in_1
){
    TYPE_INTER res[PARALLEL_DEGREE];
    for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1+= PARALLEL_DEGREE) {
        for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
            res[i_in_o] = 0;
        }
        for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
#pragma HLS PIPELINE
            int ind_in = sub2ind3(0, i_in_0, i_in_1, array_in_size_0, array_in_size_1);
            int ind_w = sub2ind3(0, i_in_0, i_w_1, array_in_size_0, array_weight_size_1);
            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
                res[i_in_o] += local[i_in_0]//array[a1_offset+ind_in]
                            * array[(a2_offset+ind_w) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
            }
        }
        int ind_out = sub2ind3(0, i_in_1, i_w_1, array_in_size_1, array_weight_size_1);
        for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
#ifdef SYNTHESIS
            grad_out[(out_offset+ind_out) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o] += res[i_in_o] >> shift;
#else 
            grad_out[(out_offset+ind_out) + i_in_o] += res[i_in_o] / pow(2, shift);
#endif
        }
    }
}

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
){
    /* tensor contraction on the first and last dimension
    ABxAE->BE
    */
#pragma HLS INTERFACE m_axi depth=1073741824 port=array offset=slave
#pragma HLS INTERFACE m_axi depth=1048576 port=grad_out
#pragma HLS ARRAY_RESHAPE variable=array cyclic factor=16
#pragma HLS ARRAY_RESHAPE variable=grad_out cyclic factor=16
#pragma HLS DEPENDENCE array inter false
    #ifndef SYNTHESIS
    assert (array_in_size_1 % PARALLEL_DEGREE == 0);
    #endif 
   
    TYPE_DATA locall[256];
    TYPE_DATA localr[256];
#pragma HLS ARRAY_RESHAPE variable=locall dim=1 factor=16
#pragma HLS ARRAY_RESHAPE variable=localr dim=1 factor=16
#pragma HLS resource variable=local core=RAM_1P
    tensor_cont_head_backward_load(
        array,
        grad_out,
        a1_offset,
        a2_offset,
        out_offset,
        array_in_size_0,
        array_in_size_1,
        array_weight_size_1,
        shift,
        localr,
        0
    );
    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
    	if (i_in_1 % 2 == 0) {
            tensor_cont_head_backward_load(
                array,
                grad_out,
                a1_offset,
                a2_offset,
                out_offset,
                array_in_size_0,
                array_in_size_1,
                array_weight_size_1,
                shift,
                locall,
                i_in_1 + 1
            );
            tensor_cont_head_backward_compute(
                array,
                grad_out,
                a1_offset,
                a2_offset,
                out_offset,
                array_in_size_0,
                array_in_size_1,
                array_weight_size_1,
                shift,
                localr,
                i_in_1
            );
        }
        else {
            tensor_cont_head_backward_load(
                array,
                grad_out,
                a1_offset,
                a2_offset,
                out_offset,
                array_in_size_0,
                array_in_size_1,
                array_weight_size_1,
                shift,
                localr,
                i_in_1 + 1
            );
            tensor_cont_head_backward_compute(
                array,
                grad_out,
                a1_offset,
                a2_offset,
                out_offset,
                array_in_size_0,
                array_in_size_1,
                array_weight_size_1,
                shift,
                locall,
                i_in_1
            );
        }
        
    }
}


//void tensor_cont_mid_wrapper(
//	TYPE_DATA array_in[1073741824],
//	TYPE_WEIGHT array_weight[1048576]
//	//TYPE_DATA array_out[1073741824]
//) {
//
//
//#pragma HLS INTERFACE m_axi depth=1073741824 port=array_in offset=slave
//#pragma HLS INTERFACE ap_memory depth=1048576 port=array_weight
//#pragma HLS ARRAY_RESHAPE variable=array_in cyclic factor=16
//#pragma HLS ARRAY_RESHAPE variable=array_weight cyclic factor=16
////#pragma HLS DEPENDENCE variable=array_in false
//		tensor_cont_mid(
//			array_in,
//			array_weight,
//			array_in + 4096*64,
//			8,
//			8,
//			32,
//			4,
//			4
//		);
//		tensor_cont_mid(
//			array_in + 256,
//			array_weight + 1024,
//			array_out + 512,
//			6,
//			6,
//			48,
//			3,
//			3
//		);
//    int array_in_size_0 = 8;
//    int array_in_size_1 = 8;
//    int array_in_size_2 = 32;
//    int array_weight_size_0 = 4;
//    int array_weight_size_2 = 4;
//
//    TYPE_INTER res[PARALLEL_DEGREE];
//    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
//        for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
//            for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2+=PARALLEL_DEGREE) {
//				for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2++) {
////#pragma HLS pipeline
//					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//#pragma HLS UNROLL
//                        res[i_in_o] = 0;
//					}
//					for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1 += 1) {
//#pragma HLS pipeline
//						int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2, array_in_size_1, array_in_size_2) / PARALLEL_DEGREE;
//						int ind_w = sub2ind3(i_w_0, i_in_1, i_w_2, array_in_size_1, array_weight_size_2);
//						for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//#pragma HLS UNROLL
//							res[i_in_o] += array_in[ind_in * PARALLEL_DEGREE + i_in_o] * array_weight[ind_w];
//						}
//					}
//					int ind_out = sub2ind4(i_in_0, i_w_0, i_w_2,  i_in_2,
//						array_weight_size_0, array_weight_size_2, array_in_size_2) / PARALLEL_DEGREE;
//					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//#pragma HLS UNROLL
//                        array_in[4096*64 + ind_out * PARALLEL_DEGREE + i_in_o] = res[i_in_o];
//                    }
//                }
//            }
//        }
//    }
//}
