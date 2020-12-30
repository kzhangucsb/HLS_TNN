#include "tt_nn.h"
#ifdef SYNTHESIS
#include <ap_fixed.h>

static TYPE_RINT lfsr = 1;

void seed(TYPE_RINT init){
    lfsr = init;
}

TYPE_RINT rand_step() {
//    bool b_32 = lfsr.get_bit(32-32);
//    bool b_22 = lfsr.get_bit(32-22);
//    bool b_2 = lfsr.get_bit(32-2);
//    bool b_1 = lfsr.get_bit(32-1);
//    bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
//    lfsr = lfsr >> 1;
//    lfsr.set_bit(31, new_bit);
//;
//    return lfsr;
	lfsr ^= lfsr << 13;
	lfsr ^= lfsr >> 7;
	lfsr ^= lfsr << 17;
	return lfsr;
}

TYPE_INTER getrand(int pos){
	pos = pos % 8;
	TYPE_INTER tmp = 0;
	tmp.range(3, 0) = lfsr.range(pos*4+3, pos*4);
	return tmp;
}
#endif
