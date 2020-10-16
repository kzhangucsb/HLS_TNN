#include <stdio.h>
#include "tt_nn.h"

void mnist_forward(
    TYPE_DATA* data,
    TYPE_WEIGHT** weight,
    TYPE_WEIGHT** bias,
    TYPE_DATA* output,
    int batchsize
);

int main(){
    TYPE_DATA* data = new TYPE_DATA[28*28*100];
    TYPE_DATA* output = new TYPE_DATA[10*100];
    TYPE_WEIGHT* weight[6];
    TYPE_DATA* bias[2];
    FILE* f;
    weight[0] = new TYPE_WEIGHT[7*4*20];
    weight[1] = new TYPE_WEIGHT[20*4*8*20];
    weight[2] = new TYPE_WEIGHT[20*7*4*20];
    weight[3] = new TYPE_WEIGHT[20*4*4];
    weight[4] = new TYPE_WEIGHT[32*2*20];
    weight[5] = new TYPE_WEIGHT[16*5*20];
    bias[0] = new TYPE_DATA[512];
    bias[1] = new TYPE_DATA[10];

    f = fopen("weight0.bin", "rb");
    fread((void*)weight[0], sizeof(TYPE_WEIGHT), 7*4*20, f);
    fclose(f);
    f = fopen("weight1.bin", "rb");
    fread((void*)weight[1], sizeof(TYPE_WEIGHT), 20*4*8*20, f);
    fclose(f);
    f = fopen("weight2.bin", "rb");
    fread((void*)weight[2], sizeof(TYPE_WEIGHT), 20*7*4*20, f);
    fclose(f);
    f = fopen("weight3.bin", "rb");
    fread((void*)weight[3], sizeof(TYPE_WEIGHT), 20*4*4, f);
    fclose(f);
    f = fopen("weight4.bin", "rb");
    fread((void*)weight[4], sizeof(TYPE_WEIGHT), 32*2*20, f);
    fclose(f);
    f = fopen("weight5.bin", "rb");
    fread((void*)weight[5], sizeof(TYPE_WEIGHT), 16*5*20, f);
    fclose(f);

    f = fopen("bias0.bin", "rb");
    fread((void*)bias[0], sizeof(TYPE_DATA), 512, f);
    fclose(f); 
    f = fopen("bias1.bin", "rb");
    fread((void*)bias[1], sizeof(TYPE_DATA), 10, f);
    fclose(f); 

    f = fopen("input.bin", "rb");
    fread((void*)data, sizeof(TYPE_DATA), 28*28*100, f);
    fclose(f);

    mnist_forward(data, weight, bias, output, 100);

    f = fopen("output.bin", "wb");
    fwrite((void*)output, sizeof(TYPE_DATA), 10*100, f);
    fclose(f);
    return 0;
}