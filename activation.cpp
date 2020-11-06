#include "tt_nn.h"

void relu_inplace(
    TYPE_DATA* data,
    int shape
){
    for (int i = 0; i < shape; i++) {
        data[i] = data[i] > TYPE_DATA(0) ? data[i] : TYPE_DATA(0);
    }
}

void relu_backward_inplace(
    TYPE_DATA* data,
    TYPE_DATA* grad,
    int shape
){
    for (int i = 0; i < shape; i++) {
        grad[i] = data[i] > TYPE_DATA(0) ? grad[i] : TYPE_DATA(0);
    }
}

