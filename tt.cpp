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
    TYPE_DATA array_in[78400],
    TYPE_WEIGHT array_weight[10000],
    TYPE_DATA array_out[78400],
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
    TYPE_INTER res;
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

