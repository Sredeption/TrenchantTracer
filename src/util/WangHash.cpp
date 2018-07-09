//
// Created by issac on 18-7-8.
//

#include "util/WangHash.h"

// this hash function calculates a new random number generator seed for each frame, based on frameNumber
unsigned int WangHash::encode(unsigned int a) {
    a = (a ^ 61u) ^ (a >> 16u);
    a = a + (a << 3u);
    a = a ^ (a >> 4u);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15u);
    return a;
}
