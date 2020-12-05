#include "tt_nn.h"
#ifndef SYNTHESIS
#include <iostream>
using namespace std;
#endif

void relu_inplace(
    TYPE_DATA* data,
    int offset,
    int shape
){
    #ifndef SYNTHESIS
    cout << "relu_inplace(array, ";
    cout << offset << ", ";
    cout << shape << ");" << endl;
    #endif
    for (int i = 0; i < shape; i++) {
        data[offset + i] = data[offset + i] > TYPE_DATA(0) ? data[offset + i] : TYPE_DATA(0);
    }
}

void relu_backward_inplace(
    TYPE_DATA* data,
    int data_offset,
    int grad_offset,
    int shape
){
    #ifndef SYNTHESIS
    cout << "relu_backward_inplace(array, ";
    cout << data_offset << ", ";
    cout << grad_offset << ", ";
    cout << shape << ");" << endl;
    #endif
    for (int i = 0; i < shape; i++) {
        data[grad_offset + i] = data[data_offset + i] > TYPE_DATA(0) ? data[grad_offset + i] : TYPE_DATA(0);
    }
}

void softmax_ce_grad(
    TYPE_DATA* data,
    unsigned char label,
    int out_offset,
    int grad_offset,
    unsigned char num_class
){
    #ifndef SYNTHESIS
    cout << "softmax_ce_grad(array, ";
    cout << "label" << ", ";
    cout << out_offset << ", ";
    cout << grad_offset << ", ";
    cout << "num_class" << ");" << endl;
    #endif
    TYPE_DATA max_val = data[out_offset];
    for (int i = 1; i < num_class; i++) {
        max_val =  max_val > data[out_offset + i] ? max_val : data[out_offset + i];
    }
    TYPE_DATA sum = 0;
    for (int i = 0; i < num_class; i++) {
        data[grad_offset + i] = exp(data[out_offset + i] - max_val);
        sum += data[grad_offset + i];
    }
    for (int i = 0; i < num_class; i++) {
        data[grad_offset + i] /= sum;
    }
    data[grad_offset + label] -= 1;
}

