#include "tt_nn.h"
#ifndef SYNTHESIS
#include <assert.h>
#include <iostream>
using namespace std;
#endif
#include <string.h>

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
	TYPE_INTER res[128][PARALLEL_DEGREE2 * PARALLEL_DEGREE],
    int i_in_0,
    int i_in_2
){
	TYPE_INTER res_t[PARALLEL_DEGREE2 * PARALLEL_DEGREE];
#pragma HLS ARRAY_PARTITION variable=res_t complete dim=1
	for (int i_w = 0; i_w < 128; i_w++) {
#pragma HLS pipeline
		for (int i_o = 0; i_o < PARALLEL_DEGREE2 * PARALLEL_DEGREE; i_o++) {
#pragma HLS UNROLL
			res[i_w][i_o] = 0;
		}
	}

    for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
        for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2+=PARALLEL_DEGREE2) {
            loop_in_1: for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1 += 1) {
#pragma HLS pipeline
                int ind_w = sub2ind3(i_w_0, i_in_1, i_w_2, array_in_size_1, array_weight_size_2);
                for (int i_o = 0; i_o < PARALLEL_DEGREE2 * PARALLEL_DEGREE; i_o++){
                	int i_in_o = i_o % PARALLEL_DEGREE;
                	int i_in_o2 = i_o / PARALLEL_DEGREE;
						res_t[i_o]
						+= local[i_in_1][i_in_o]
						* TYPE_DATA(weight[(weight_offset + ind_w) / PARALLEL_DEGREE2 * PARALLEL_DEGREE2 + i_in_o2]);
                }
            }
            for (int i_o = 0; i_o < PARALLEL_DEGREE2 * PARALLEL_DEGREE; i_o++){
            	res[(i_w_0 * array_weight_size_2 + i_w_2) / PARALLEL_DEGREE2][i_o] = res_t[i_o]  >> shift;
            }

        }
    }
}

void tensor_cont_mid_store(
    TYPE_DATA array[1073741824],
    int in_offset,
    int out_offset,
    int weight_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2,
    int shift,
	TYPE_INTER res[128][PARALLEL_DEGREE2 * PARALLEL_DEGREE],
    int i_in_0,
    int i_in_2
){
	for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
	    for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2++) {
#pragma HLS pipeline
	    	int ind_out = sub2ind4(i_in_0, i_w_0, i_w_2,  i_in_2,
	    			array_weight_size_0, array_weight_size_2, array_in_size_2);
	    	rand_step();

			for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
#pragma HLS UNROLL
				array[(out_offset + ind_out) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
					  = res[i_w_2 / PARALLEL_DEGREE2][(i_w_2 % PARALLEL_DEGREE2) * PARALLEL_DEGREE + i_in_o] + getrand(i_in_o);
			}
	    }
	}
//	for (int i_w = 0; i_w < 128; i_w++) {
//#pragma HLS pipeline
//		for (int i_o = 0; i_o < PARALLEL_DEGREE2 * PARALLEL_DEGREE; i_o++) {
//#pragma HLS UNROLL
//			res[i_w][i_o] = 0;
//		}
//	}
}

void tensor_cont_mid_load(
    TYPE_DATA array[1073741824],
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
	TYPE_DATA array2[1073741824],
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
	assert (array_weight_size_2 % PARALLEL_DEGREE == 0);
    assert (array_in_size_2 % PARALLEL_DEGREE == 0);
    #endif 
    //TYPE_INTER res[PARALLEL_DEGREE];
//    TYPE_DATA locall[1024][PARALLEL_DEGREE];
//    TYPE_DATA localr[1024][PARALLEL_DEGREE];
//#pragma HLS ARRAY_RESHAPE variable=locall dim=2
//#pragma HLS ARRAY_RESHAPE variable=localr dim=2
//#pragma HLS resource variable=locall core=RAM_1P
//#pragma HLS resource variable=localr core=RAM_1P
//#pragma HLS ARRAY_RESHAPE variable=locall dim=2
//#pragma HLS ARRAY_RESHAPE variable=localr dim=2
//    TYPE_INTER resl[64][PARALLEL_DEGREE2 * PARALLEL_DEGREE];
//    TYPE_INTER resr[64][PARALLEL_DEGREE2 * PARALLEL_DEGREE];
//#pragma HLS ARRAY_RESHAPE variable=resl dim=2 complete
//#pragma HLS ARRAY_RESHAPE variable=resr dim=2 complete
//#pragma HLS RESOURCE variable=resl core=RAM_1P_LUTRAM
//#pragma HLS RESOURCE variable=resr core=RAM_1P_LUTRAM
    for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2+=PARALLEL_DEGREE) {
//        tensor_cont_mid_load(
//            array, in_offset, out_offset, weight_offset,
//            array_in_size_0, array_in_size_1, array_in_size_2,
//            array_weight_size_0, array_weight_size_2, shift,
//            localr, 0, i_in_2
//        );
        for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
#pragma HLS dataflow
        	TYPE_DATA local[1024][PARALLEL_DEGREE];
#pragma HLS ARRAY_RESHAPE variable=local dim=2
#pragma HLS resource variable=local core=RAM_1P
			TYPE_INTER res[64][PARALLEL_DEGREE2 * PARALLEL_DEGREE];
#pragma HLS ARRAY_RESHAPE variable=res dim=2 complete
#pragma HLS RESOURCE variable=res core=RAM_1P_LUTRAM

        	tensor_cont_mid_load(
				array, in_offset, out_offset, weight_offset,
				array_in_size_0, array_in_size_1, array_in_size_2,
				array_weight_size_0, array_weight_size_2, shift,
				local, i_in_0, i_in_2
			);
        	tensor_cont_mid_compute(
				weight, in_offset, out_offset, weight_offset,
				array_in_size_0, array_in_size_1, array_in_size_2,
				array_weight_size_0, array_weight_size_2, shift,
				local, res, i_in_0, i_in_2
			);
			tensor_cont_mid_store(
				array2, in_offset, out_offset, weight_offset,
				array_in_size_0, array_in_size_1, array_in_size_2,
				array_weight_size_0, array_weight_size_2, shift,
				res, i_in_0, i_in_2
			);
        }
    }
}
//            if (i_in_0 % 2 == 0){
//                tensor_cont_mid_load(
//                    array, in_offset, out_offset, weight_offset,
//                    array_in_size_0, array_in_size_1, array_in_size_2,
//                    array_weight_size_0, array_weight_size_2, shift,
//                    locall, i_in_0 + 1, i_in_2
//                );
//                if (i_in_0 > 0) {
//					tensor_cont_mid_store(
//						array, in_offset, out_offset, weight_offset,
//						array_in_size_0, array_in_size_1, array_in_size_2,
//						array_weight_size_0, array_weight_size_2, shift,
//						resl, i_in_0 - 1, i_in_2
//					);
//				}
//                tensor_cont_mid_compute(
//                    weight, in_offset, out_offset, weight_offset,
//                    array_in_size_0, array_in_size_1, array_in_size_2,
//                    array_weight_size_0, array_weight_size_2, shift,
//                    localr, resr,  i_in_0, i_in_2
//                );
//            }
//
//            else {
//                tensor_cont_mid_load(
//                    array, in_offset, out_offset, weight_offset,
//                    array_in_size_0, array_in_size_1, array_in_size_2,
//                    array_weight_size_0, array_weight_size_2, shift,
//                    localr, i_in_0 + 1, i_in_2
//                );
//                tensor_cont_mid_store(
//					array, in_offset, out_offset, weight_offset,
//					array_in_size_0, array_in_size_1, array_in_size_2,
//					array_weight_size_0, array_weight_size_2, shift,
//					resr,  i_in_0 - 1, i_in_2
//				);
//                tensor_cont_mid_compute(
//                    weight, in_offset, out_offset, weight_offset,
//                    array_in_size_0, array_in_size_1, array_in_size_2,
//                    array_weight_size_0, array_weight_size_2, shift,
//                    locall, resl, i_in_0, i_in_2
//                );
//            }
//        }
//        if (array_in_size_0 % 2 == 0) {
//            tensor_cont_mid_store(
//				array, in_offset, out_offset, weight_offset,
//				array_in_size_0, array_in_size_1, array_in_size_2,
//				array_weight_size_0, array_weight_size_2, shift,
//				resl,  array_in_size_0 - 1, i_in_2
//			);
//        }
//        else {
//        	tensor_cont_mid_store(
//				array, in_offset, out_offset, weight_offset,
//				array_in_size_0, array_in_size_1, array_in_size_2,
//				array_weight_size_0, array_weight_size_2, shift,
//				resr,  array_in_size_0 - 1, i_in_2
//			);
//        }
//    }
//
//}

void tensor_cont_last_load(
    TYPE_DATA array[1073741824],
    int in_offset,
    int out_offset,
    int weight_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_1,
	int shift,
    TYPE_DATA local[PARALLEL_DEGREE2][32][16],
    int i_in_0
){
#pragma HLS DEPENDENCE variable=local inter false
//#pragma HLS ARRAY_PARTITION variable=local dim=3 factor=16
//#pragma HLS resource variable=local core=RAM_1P

	for (int i_in_0o = 0; i_in_0o < PARALLEL_DEGREE2; i_in_0o++){
		for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2 += PARALLEL_DEGREE) {
			for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
#pragma HLS PIPELINE
				int ind_in = sub2ind3(i_in_0 + i_in_0o, i_in_1, i_in_2,  array_in_size_1, array_in_size_2);
//				memcpy(
//					local[i_in_0o][i_in_1],
//					array + (in_offset + ind_in) / PARALLEL_DEGREE * PARALLEL_DEGREE,
//					PARALLEL_DEGREE * 2);
				for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
#pragma HLS UNROLL
					local[i_in_0o][i_in_1][i_in_2 + i_in_o]
									= array[(in_offset + ind_in) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
				}
			}
		}
	}
}

void tensor_cont_last_compute(
    TYPE_WEIGHT weight[1048576],
    int in_offset,
    int out_offset,
    int weight_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_1,
	int shift,
    TYPE_DATA local[PARALLEL_DEGREE2][32][16],
	TYPE_INTER res[PARALLEL_DEGREE2][1024],
    int i_in_0
){
#pragma HLS DEPENDENCE array false
	for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1++) {
        for (int i_in_0o = 0; i_in_0o < PARALLEL_DEGREE2; i_in_0o++) {
#pragma HLS UNROLL
        	res[i_in_0o][i_w_1] = 0;
        }
		loop_in_2: for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2 += PARALLEL_DEGREE){
			loop_in_1: for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
#pragma HLS PIPELINE
                int ind_w = sub2ind3(i_in_1, i_w_1, i_in_2, array_weight_size_1, array_in_size_2);
                for (int i_in_0o = 0; i_in_0o < PARALLEL_DEGREE2; i_in_0o++) {
#pragma HLS UNROLL
					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
						TYPE_DATA w = weight[(weight_offset+ind_w)/ PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
#pragma HLS UNROLL

						res[i_in_0o][i_w_1] += local[i_in_0o][i_in_1][i_in_2 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o] * w;
									//* TYPE_DATA(weight[(weight_offset+ind_w)/ PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]);

					}
                }
            }
        }


    }
}

void tensor_cont_last_store(
    TYPE_DATA array[1073741824],
    int in_offset,
    int out_offset,
    int weight_offset,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_1,
	int shift,
	TYPE_INTER res[PARALLEL_DEGREE2][1024],
    int i_in_0
){
        //array[out_offset + ind_out] = res >> shift;

TYPE_DATA tmp[PARALLEL_DEGREE2][PARALLEL_DEGREE];
#pragma HLS ARRAY_PARTITION variable=tmp dim=0 complete
	for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1+=PARALLEL_DEGREE) {
		for (int i_w_o = 0; i_w_o < PARALLEL_DEGREE; i_w_o++){
#pragma HLS pipeline
			rand_step();
			for (int i_in_0o = 0; i_in_0o < PARALLEL_DEGREE2; i_in_0o++) {
#pragma HLS UNROLL
				tmp[i_in_0o][i_w_o]
					  = res[i_in_0o][i_w_1 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_w_o] + getrand(i_in_0o);
			}
		}
		for (int i_in_0o = 0; i_in_0o < PARALLEL_DEGREE2; i_in_0o++) {
#pragma HLS pipeline
			for (int i_w_o = 0; i_w_o < PARALLEL_DEGREE; i_w_o++){
#pragma HLS UNROLL
				array[(out_offset + (i_in_0 + i_in_0o) * array_weight_size_1 + i_w_1) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_w_o]
					  = tmp[i_in_0o][i_w_o];
			}
		}
	}

//	for (int i_in_0o = 0; i_in_0o < PARALLEL_DEGREE2; i_in_0o++) {
//		for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1+=PARALLEL_DEGREE) {
//#pragma HLS pipeline
//			rand_step();
//			for (int i_w_o = 0; i_w_o < PARALLEL_DEGREE; i_w_o++){
//#pragma HLS UNROLL
//				array[(out_offset + (i_in_0 + i_in_0o) * array_weight_size_1 + i_w_1) / PARALLEL_DEGREE * PARALLEL_DEGREE
//						 + i_w_o]
//					  = res[i_in_0o][i_w_1 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_w_o] + getrand(i_w_o);
//			}
//		}
//	}

}

void tensor_cont_last(
    TYPE_DATA array[1073741824],
	TYPE_DATA array2[1073741824],
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
#pragma HLS DEPENDENCE array intra false
#pragma HLS DEPENDENCE array inter false
    #ifndef SYNTHESIS
    assert (array_in_size_2 == PARALLEL_DEGREE);
    #endif 
    //TYPE_INTER res;

//    TYPE_DATA locall[PARALLEL_DEGREE2][32][16];
//    TYPE_DATA localr[PARALLEL_DEGREE2][32][16];
//#pragma HLS ARRAY_RESHAPE variable=locall dim=3 factor=16
//#pragma HLS ARRAY_RESHAPE variable=locall dim=1
//#pragma HLS RESOURCE variable=locall core=RAM_1P_LUTRAM
//#pragma HLS ARRAY_RESHAPE variable=localr dim=3 factor=16
//#pragma HLS ARRAY_RESHAPE variable=localr dim=1
//#pragma HLS RESOURCE variable=localr core=RAM_1P_LUTRAM
////#pragma HLS resource variable=locall core=RAM_1P
////#pragma HLS resource variable=localr core=RAM_1P
//
//    TYPE_INTER resl[PARALLEL_DEGREE2][1024];
//    TYPE_INTER resr[PARALLEL_DEGREE2][1024];
//#pragma HLS ARRAY_PARTITION variable=resl complete dim=1
//#pragma HLS ARRAY_PARTITION variable=resr complete dim=1

//    tensor_cont_last_load(
//        array, in_offset, out_offset, weight_offset,
//        array_in_size_0, array_in_size_1, array_in_size_2,
//        array_weight_size_1, shift,
//        localr, 0
//    );
    //for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0+=PARALLEL_DEGREE2) {
    for (int i_in_base = 0; i_in_base < array_in_size_0 / PARALLEL_DEGREE2; i_in_base++){
//#pragma HLS dataflow
    	TYPE_DATA local[PARALLEL_DEGREE2][32][16];
#pragma HLS ARRAY_RESHAPE variable=local dim=3 factor=16
#pragma HLS ARRAY_RESHAPE variable=local dim=1
#pragma HLS RESOURCE variable=local core=RAM_1P_LUTRAM
    	TYPE_INTER res[PARALLEL_DEGREE2][1024];
//#pragma HLS ARRAY_RESHAPE variable=res factor=16 dim=2
#pragma HLS ARRAY_RESHAPE variable=res complete dim=1
//#pragma HLS RESOURCE variable=res core=RAM_2P_LUTRAM
    	int i_in_0 = i_in_base * PARALLEL_DEGREE2;
    	tensor_cont_last_load(
			array, in_offset, out_offset, weight_offset,
			array_in_size_0, array_in_size_1, array_in_size_2,
			array_weight_size_1, shift,
			local, i_in_0
		);
		tensor_cont_last_compute(
			weight, in_offset, out_offset, weight_offset,
			array_in_size_0, array_in_size_1, array_in_size_2,
			array_weight_size_1, shift,
			local, res, i_in_0
		);
		tensor_cont_last_store(
			array2, in_offset, out_offset, weight_offset,
			array_in_size_0, array_in_size_1, array_in_size_2,
			array_weight_size_1, shift,
			res, i_in_0
		);
    }
}
//    	//load
//        if ((i_in_0 / PARALLEL_DEGREE2) % 2 == 0) {
//            tensor_cont_last_load(
//                array, in_offset, out_offset, weight_offset,
//                array_in_size_0, array_in_size_1, array_in_size_2,
//                array_weight_size_1, shift,
//				locall, i_in_0 + 1
//            );
//            if (i_in_0 > 0){
//            	tensor_cont_last_store(
//            		array, in_offset, out_offset, weight_offset,
//					array_in_size_0, array_in_size_1, array_in_size_2,
//					array_weight_size_1, shift,
//					resl, i_in_0 - 1
//				);
//            }
//            tensor_cont_last_compute(
//            	weight, in_offset, out_offset, weight_offset,
//				array_in_size_0, array_in_size_1, array_in_size_2,
//				array_weight_size_1, shift,
//                localr, resr, i_in_0
//            );
//        }
//        else {
//            tensor_cont_last_load(
//            	array, in_offset, out_offset, weight_offset,
//				array_in_size_0, array_in_size_1, array_in_size_2,
//				array_weight_size_1, shift,
//                localr, i_in_0 + 1
//            );
//            tensor_cont_last_store(
//            	array, in_offset, out_offset, weight_offset,
//				array_in_size_0, array_in_size_1, array_in_size_2,
//				array_weight_size_1, shift,
//				resr, i_in_0 - 1
//			);
//            tensor_cont_last_compute(
//            	weight, in_offset, out_offset, weight_offset,
//            	array_in_size_0, array_in_size_1, array_in_size_2,
//            	array_weight_size_1, shift,
//                locall, resl, i_in_0
//            );
//        }
//    }
//    if (array_in_size_0 % 2 == 0) {
//    	tensor_cont_last_store(
//    		array, in_offset, out_offset, weight_offset,
//    		array_in_size_0, array_in_size_1, array_in_size_2,
//    		array_weight_size_1, shift,
//			resl, array_in_size_0 - 1
//		);
//    }
//    else {
//    	tensor_cont_last_store(
//    		array, in_offset, out_offset, weight_offset,
//    		array_in_size_0, array_in_size_1, array_in_size_2,
//    		array_weight_size_1, shift,
//    		resr, array_in_size_0 - 1
//    	);
//    }
//}

void tensor_cont_outer_prod(
	const TYPE_DATA array1[1073741824],
	TYPE_DATA array2[1073741824],
	int data_in_offset,
	int grad_out_offset,
	int grad_wt_offset,
	const int stride1[5],
	const int stride2[5],
	const int strideo[10],
	const int shape1exp[5],
	const int shape2exp[5],
    int shift
){
#pragma HLS DEPENDENCE variable=array2 inter false
#pragma HLS DEPENDENCE variable=array2 intra false

#pragma HLS ARRAY_PARTITION variable=stride1 complete dim=1
#pragma HLS ARRAY_PARTITION variable=stride2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=strideo complete dim=1
#pragma HLS ARRAY_PARTITION variable=shape1exp complete dim=1
#pragma HLS ARRAY_PARTITION variable=shape2exp complete dim=1
    int ind_i = 0, ind_o = 0, ind_r = 0;
    int index1[5], index2[5];
#pragma HLS ARRAY_PARTITION variable=index2 complete dim=1
#pragma HLS ARRAY_PARTITION variable=index1 complete dim=1
    for (index2[0] = 0; index2[0] < shape2exp[0]; index2[0]++) {
    for (index2[1] = 0; index2[1] < shape2exp[1]; index2[1]++) {
    for (index2[2] = 0; index2[2] < shape2exp[2]; index2[2]++) {
    for (index2[3] = 0; index2[3] < shape2exp[3]; index2[3]++) {
    for (index2[4] = 0; index2[4] < shape2exp[4]; index2[4]++) {
    	TYPE_DATA go = array1[grad_out_offset + ind_o];
    for (index1[0] = 0; index1[0] < shape1exp[0]; index1[0]++) {
    for (index1[1] = 0; index1[1] < shape1exp[1]; index1[1]++) {
    for (index1[2] = 0; index1[2] < shape1exp[2]; index1[2]++) {
    for (index1[3] = 0; index1[3] < shape1exp[3]; index1[3]++) {
#pragma HLS pipeline
    	rand_step();
    for (int i = 0; i < shape1exp[4]; i++) {
#pragma HLS UNROLL skip_exit_check factor=16
        TYPE_INTER res = array1[(data_in_offset + ind_i) / PARALLEL_DEGREE * PARALLEL_DEGREE + i] * go;
        res = res >> shift;
        array2[(grad_wt_offset + ind_r) / PARALLEL_DEGREE * PARALLEL_DEGREE + i] += TYPE_GRAD(res + getrand(i));

//        ind_i += 1;
//        ind_r += 1;
    }
//        ind_i -= stride1[4] * shape1exp[4];
//        ind_r -= strideo[9] * shape1exp[4];
        ind_i += stride1[3];
        ind_r += strideo[7];
    }
        ind_i -= stride1[3] * shape1exp[3];
        ind_r -= strideo[7] * shape1exp[3];
        ind_i += stride1[2];
        ind_r += strideo[5];
    }
        ind_i -= stride1[2] * shape1exp[2];
        ind_r -= strideo[5] * shape1exp[2];
        ind_i += stride1[1];
        ind_r += strideo[3];
    }
        ind_i -= stride1[1] * shape1exp[1];
        ind_r -= strideo[3] * shape1exp[1];
        ind_i += stride1[0];
        ind_r += strideo[1];
    }
        ind_i = 0;
        ind_r -= strideo[1];
        ind_o += stride2[4];
        ind_r += strideo[8];
    }
        ind_o -= stride2[4] * shape2exp[4];
        ind_r -= strideo[8] * shape2exp[4];
        ind_o += stride2[3];
        ind_r += strideo[6];
    }
        ind_o -= stride2[3] * shape2exp[3];
        ind_r -= strideo[6] * shape2exp[3];
        ind_o += stride2[2];
        ind_r += strideo[4];
    }
        ind_o -= stride2[2] * shape2exp[2];
        ind_r -= strideo[4] * shape2exp[2];
        ind_o += stride2[1];
        ind_r += strideo[2];
    }
        ind_o -= stride2[1] * shape2exp[1];
        ind_r -= strideo[2] * shape2exp[1];
        ind_o += stride2[0];
        ind_r += strideo[0];
    }
}

//void tensor_cont_end_backward_load(
//    TYPE_DATA array[1073741824],
//    TYPE_GRAD grad_out[1073741824],
//    int a1_offset,
//    int a2_offset,
//    int out_offset,
//    int array_in_size_0,
//    int array_in_size_1,
//    int array_in_size_2,
//    int array_in_size_3,
//    int array_weight_size_1,
//	int shift,
//    TYPE_DATA local[32][128],
//    int i_in_1,
//    int i_in_2
//)
//{
//    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
//        for (int i_in_3 = 0; i_in_3 < array_in_size_3; i_in_3 += PARALLEL_DEGREE) {
//#pragma HLS PIPELINE
//            int ind_in = sub2ind4(i_in_0, i_in_1, i_in_2, i_in_3,
//                    array_in_size_1, array_in_size_2, array_in_size_3);
//            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
//#pragma HLS UNROLL
//                local[i_in_0][i_in_3 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
//                            = array[(a1_offset+ind_in) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
//            }
//        }
//    }
//}
//void tensor_cont_end_backward_compute(
//    TYPE_DATA array[1073741824],
//    TYPE_GRAD grad_out[1073741824],
//    int a1_offset,
//    int a2_offset,
//    int out_offset,
//    int array_in_size_0,
//    int array_in_size_1,
//    int array_in_size_2,
//    int array_in_size_3,
//    int array_weight_size_1,
//	int shift,
//    TYPE_DATA local[32][128],
//    int i_in_1,
//    int i_in_2
//){
//    TYPE_INTER res;
//    for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1++) {
//        res = 0;
//        for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
//            for (int i_in_3 = 0; i_in_3 < array_in_size_3; i_in_3 += PARALLEL_DEGREE) {
//#pragma HLS PIPELINE
////                        int ind_in = sub2ind4(i_in_0, i_in_1, i_in_2, i_in_3,
////                            array_in_size_1, array_in_size_2, array_in_size_3);
//                int ind_w = sub2ind3(i_in_0, i_w_1, i_in_3, array_weight_size_1, array_in_size_3);
//                for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++) {
//#pragma HLS UNROLL
//                    //res += array[(a1_offset+ind_in) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
//                    res += local[i_in_0][i_in_3 / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o]
//                                    * array[(a2_offset+ind_w) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
//                }
//            }
//        }
//        int ind_out = sub2ind3(i_in_1, i_w_1, i_in_2, array_weight_size_1, array_in_size_2);
//#ifdef SYNTHESIS
//        grad_out[out_offset+ind_out] += res >> shift;
//#else
//        grad_out[out_offset+ind_out] += res / pow(2, shift);
//#endif
//    }
//
//}
//
//void tensor_cont_end_backward(
//    TYPE_DATA array[1073741824],
//    TYPE_GRAD grad_out[1073741824],
//    int a1_offset,
//    int a2_offset,
//    int out_offset,
//    int array_in_size_0,
//    int array_in_size_1,
//    int array_in_size_2,
//    int array_in_size_3,
//    int array_weight_size_1,
//	int shift
//){
//    /* tensor contraction on the first and last dimension
//    ABCDxAED->BEC
//    */
//#pragma HLS INTERFACE m_axi depth=1073741824 port=array offset=slave
//#pragma HLS INTERFACE m_axi depth=1048576 port=grad_out
//#pragma HLS ARRAY_RESHAPE variable=array cyclic factor=16
//#pragma HLS ARRAY_RESHAPE variable=grad_out cyclic factor=16
//#pragma HLS DEPENDENCE array inter false
//    #ifndef SYNTHESIS
//    assert (array_in_size_3 % PARALLEL_DEGREE == 0);
//    #endif
//
//    TYPE_DATA locall[32][128];
//    TYPE_DATA localr[32][128];
//#pragma HLS ARRAY_PARTITION variable=locall dim=2 factor=16
//#pragma HLS ARRAY_PARTITION variable=localr dim=2 factor=16
//#pragma HLS resource variable=locall core=RAM_1P
//#pragma HLS resource variable=localr core=RAM_1P
//    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
//        tensor_cont_end_backward_load(
//            array,
//            grad_out,
//            a1_offset,
//            a2_offset,
//            out_offset,
//            array_in_size_0,
//            array_in_size_1,
//            array_in_size_2,
//            array_in_size_3,
//            array_weight_size_1,
//            shift,
//            localr,
//            i_in_1,
//            0
//        );
//        for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2++) {
//        	if (i_in_2 % 2 == 0) {
//                tensor_cont_end_backward_load(
//                    array,
//                    grad_out,
//                    a1_offset,
//                    a2_offset,
//                    out_offset,
//                    array_in_size_0,
//                    array_in_size_1,
//                    array_in_size_2,
//                    array_in_size_3,
//                    array_weight_size_1,
//                    shift,
//                    locall,
//                    i_in_1,
//                    i_in_2 + 1
//                );
//                tensor_cont_end_backward_compute(
//                    array,
//                    grad_out,
//                    a1_offset,
//                    a2_offset,
//                    out_offset,
//                    array_in_size_0,
//                    array_in_size_1,
//                    array_in_size_2,
//                    array_in_size_3,
//                    array_weight_size_1,
//                    shift,
//                    localr,
//                    i_in_1,
//                    i_in_2
//                );
//            }
//            else {
//                tensor_cont_end_backward_load(
//                    array,
//                    grad_out,
//                    a1_offset,
//                    a2_offset,
//                    out_offset,
//                    array_in_size_0,
//                    array_in_size_1,
//                    array_in_size_2,
//                    array_in_size_3,
//                    array_weight_size_1,
//                    shift,
//                    localr,
//                    i_in_1,
//                    i_in_2 + 1
//                );
//                tensor_cont_end_backward_compute(
//                    array,
//                    grad_out,
//                    a1_offset,
//                    a2_offset,
//                    out_offset,
//                    array_in_size_0,
//                    array_in_size_1,
//                    array_in_size_2,
//                    array_in_size_3,
//                    array_weight_size_1,
//                    shift,
//                    locall,
//                    i_in_1,
//                    i_in_2
//                );
//            }
//        }
//    }
//}
//
//void tensor_cont_head_backward_load(
//    TYPE_DATA array[1073741824],
//    TYPE_GRAD grad_out[1073741824],
//    int a1_offset,
//    int a2_offset,
//    int out_offset,
//    int array_in_size_0,
//    int array_in_size_1,
//    int array_weight_size_1,
//	int shift,
//    TYPE_DATA local[256],
//    int i_in_1
//){
//    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
//#pragma HLS PIPELINE
//        int ind_in = sub2ind3(0, i_in_0, i_in_1, array_in_size_0, array_in_size_1);
//        local[i_in_0] = array[a1_offset+ind_in];
//    }
//}
//
//void tensor_cont_head_backward_compute(
//    TYPE_DATA array[1073741824],
//    TYPE_GRAD grad_out[1073741824],
//    int a1_offset,
//    int a2_offset,
//    int out_offset,
//    int array_in_size_0,
//    int array_in_size_1,
//    int array_weight_size_1,
//	int shift,
//    TYPE_DATA local[256],
//    int i_in_1
//){
//    TYPE_INTER res[PARALLEL_DEGREE];
//    for (int i_w_1 = 0; i_w_1 < array_weight_size_1; i_w_1+= PARALLEL_DEGREE) {
//        for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//#pragma HLS UNROLL
//            res[i_in_o] = 0;
//        }
//        for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
//#pragma HLS PIPELINE
//            int ind_in = sub2ind3(0, i_in_0, i_in_1, array_in_size_0, array_in_size_1);
//            int ind_w = sub2ind3(0, i_in_0, i_w_1, array_in_size_0, array_weight_size_1);
//            for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//#pragma HLS UNROLL
//                res[i_in_o] += local[i_in_0]//array[a1_offset+ind_in]
//                            * array[(a2_offset+ind_w) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o];
//            }
//        }
//        int ind_out = sub2ind3(0, i_in_1, i_w_1, array_in_size_1, array_weight_size_1);
//        for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
//#pragma HLS UNROLL
//#ifdef SYNTHESIS
//            grad_out[(out_offset+ind_out) / PARALLEL_DEGREE * PARALLEL_DEGREE + i_in_o] += res[i_in_o] >> shift;
//#else
//            grad_out[(out_offset+ind_out) + i_in_o] += res[i_in_o] / pow(2, shift);
//#endif
//        }
//    }
//}
//
//void tensor_cont_head_backward(
//    TYPE_DATA array[1073741824],
//    TYPE_GRAD grad_out[1073741824],
//    int a1_offset,
//    int a2_offset,
//    int out_offset,
//    int array_in_size_0,
//    int array_in_size_1,
//    int array_weight_size_1,
//	int shift
//){
//    /* tensor contraction on the first and last dimension
//    ABxAE->BE
//    */
//#pragma HLS INTERFACE m_axi depth=1073741824 port=array offset=slave
//#pragma HLS INTERFACE m_axi depth=1048576 port=grad_out
//#pragma HLS ARRAY_RESHAPE variable=array cyclic factor=16
//#pragma HLS ARRAY_RESHAPE variable=grad_out cyclic factor=16
//#pragma HLS DEPENDENCE array inter false
//    #ifndef SYNTHESIS
//    assert (array_in_size_1 % PARALLEL_DEGREE == 0);
//    #endif
//
//    TYPE_DATA locall[256];
//    TYPE_DATA localr[256];
//#pragma HLS ARRAY_RESHAPE variable=locall dim=1 factor=16
//#pragma HLS ARRAY_RESHAPE variable=localr dim=1 factor=16
//#pragma HLS resource variable=locall core=RAM_1P
//#pragma HLS resource variable=localr core=RAM_1P
//    tensor_cont_head_backward_load(
//        array,
//        grad_out,
//        a1_offset,
//        a2_offset,
//        out_offset,
//        array_in_size_0,
//        array_in_size_1,
//        array_weight_size_1,
//        shift,
//        localr,
//        0
//    );
//    for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1++) {
//    	if (i_in_1 % 2 == 0) {
//            tensor_cont_head_backward_load(
//                array,
//                grad_out,
//                a1_offset,
//                a2_offset,
//                out_offset,
//                array_in_size_0,
//                array_in_size_1,
//                array_weight_size_1,
//                shift,
//                locall,
//                i_in_1 + 1
//            );
//            tensor_cont_head_backward_compute(
//                array,
//                grad_out,
//                a1_offset,
//                a2_offset,
//                out_offset,
//                array_in_size_0,
//                array_in_size_1,
//                array_weight_size_1,
//                shift,
//                localr,
//                i_in_1
//            );
//        }
//        else {
//            tensor_cont_head_backward_load(
//                array,
//                grad_out,
//                a1_offset,
//                a2_offset,
//                out_offset,
//                array_in_size_0,
//                array_in_size_1,
//                array_weight_size_1,
//                shift,
//                localr,
//                i_in_1 + 1
//            );
//            tensor_cont_head_backward_compute(
//                array,
//                grad_out,
//                a1_offset,
//                a2_offset,
//                out_offset,
//                array_in_size_0,
//                array_in_size_1,
//                array_weight_size_1,
//                shift,
//                locall,
//                i_in_1
//            );
//        }
//
//    }
//}
//
//
////void tensor_cont_mid_wrapper(
////	TYPE_DATA array_in[1073741824],
////	TYPE_WEIGHT array_weight[1048576]
////	//TYPE_DATA array_out[1073741824]
////) {
////
////
////#pragma HLS INTERFACE m_axi depth=1073741824 port=array_in offset=slave
////#pragma HLS INTERFACE ap_memory depth=1048576 port=array_weight
////#pragma HLS ARRAY_RESHAPE variable=array_in cyclic factor=16
////#pragma HLS ARRAY_RESHAPE variable=array_weight cyclic factor=16
//////#pragma HLS DEPENDENCE variable=array_in false
////		tensor_cont_mid(
////			array_in,
////			array_weight,
////			array_in + 4096*64,
////			8,
////			8,
////			32,
////			4,
////			4
////		);
////		tensor_cont_mid(
////			array_in + 256,
////			array_weight + 1024,
////			array_out + 512,
////			6,
////			6,
////			48,
////			3,
////			3
////		);
////    int array_in_size_0 = 8;
////    int array_in_size_1 = 8;
////    int array_in_size_2 = 32;
////    int array_weight_size_0 = 4;
////    int array_weight_size_2 = 4;
////
////    TYPE_INTER res[PARALLEL_DEGREE];
////    for (int i_in_0 = 0; i_in_0 < array_in_size_0; i_in_0++) {
////        for (int i_w_0 = 0; i_w_0 < array_weight_size_0; i_w_0++) {
////            for (int i_in_2 = 0; i_in_2 < array_in_size_2; i_in_2+=PARALLEL_DEGREE) {
////				for (int i_w_2 = 0; i_w_2 < array_weight_size_2; i_w_2++) {
//////#pragma HLS pipeline
////					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
////#pragma HLS UNROLL
////                        res[i_in_o] = 0;
////					}
////					for (int i_in_1 = 0; i_in_1 < array_in_size_1; i_in_1 += 1) {
////#pragma HLS pipeline
////						int ind_in = sub2ind3(i_in_0, i_in_1, i_in_2, array_in_size_1, array_in_size_2) / PARALLEL_DEGREE;
////						int ind_w = sub2ind3(i_w_0, i_in_1, i_w_2, array_in_size_1, array_weight_size_2);
////						for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
////#pragma HLS UNROLL
////							res[i_in_o] += array_in[ind_in * PARALLEL_DEGREE + i_in_o] * array_weight[ind_w];
////						}
////					}
////					int ind_out = sub2ind4(i_in_0, i_w_0, i_w_2,  i_in_2,
////						array_weight_size_0, array_weight_size_2, array_in_size_2) / PARALLEL_DEGREE;
////					for (int i_in_o = 0; i_in_o < PARALLEL_DEGREE; i_in_o++){
////#pragma HLS UNROLL
////                        array_in[4096*64 + ind_out * PARALLEL_DEGREE + i_in_o] = res[i_in_o];
////                    }
////                }
////            }
////        }
////    }
////}
