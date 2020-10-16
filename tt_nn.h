#define TYPE_WEIGHT float
#define TYPE_DATA float

void tensor_contraction_raw(
    TYPE_DATA* array_in,
    TYPE_WEIGHT* array_weight,
    TYPE_DATA* array_out,
    int array_in_size_0,
    int array_in_size_1,
    int array_in_size_2,
    int array_weight_size_0,
    int array_weight_size_2
);

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
);

void relu_inplace(
    TYPE_DATA* data,
    int shape
);