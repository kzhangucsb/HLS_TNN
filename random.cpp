#include "tt_nn.h"
#ifdef SYNTHESIS
#include <ap_fixed.h>
#define RINT_TYPE ap_uint<32>

static RINT_TYPE lfsr = 1;

void seed(RINT_TYPE init){
    lfsr = init;
}

RINT_TYPE pseudo_random() {
    bool b_32 = lfsr.get_bit(32-32);
    bool b_22 = lfsr.get_bit(32-22);
    bool b_2 = lfsr.get_bit(32-2);
    bool b_1 = lfsr.get_bit(32-1);
    bool new_bit = b_32 ^ b_22 ^ b_2 ^ b_1;
    lfsr = lfsr >> 1;
    lfsr.set_bit(31, new_bit);

    return lfsr;

}
#endif
