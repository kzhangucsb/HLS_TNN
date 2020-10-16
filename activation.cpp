#include "tt_nn.h"

void relu_inplace(
    TYPE_DATA* data,
    int shape
){
    for (int i = 0; i < shape; i++) {
        data[i] = data[i] > 0 ? data[i] : 0;
    }
}

