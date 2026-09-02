/*
    Initializer:
    CBRNG_Random.h 
 
    Header file for a random number generator based on the
    D.E. Shaw counter-based RNGs as described in the paper
    Salmon et al. (2011) "Parallel Random Numbers: As Easy 
    as 1, 2, 3", Proceedings of the International Conference 
    for High Performance Computing, Networking, Storage and 
    Analysis (SC11). 

                    JD Emberson, March 2016
                       jemberson@anl.gov
 */

#ifndef CBRNG_Random_Header_Included
#define CBRNG_Random_Header_Included

// Here is where we choose our counter-based RNG
#include <Random123/threefry.h>
typedef r123::Threefry2x64 RNG;

// Some definitions based on this CBRNG
typedef RNG::ctr_type ctr_type;
typedef RNG::key_type key_type;
const int nsize_r123 = ctr_type::static_size;

void GetSlabKeys(unsigned long *keys, int x1, int numx, unsigned long seed);
void GetRandomDoublesWhiteNoise(double &rn1, double &rn2, unsigned long ikey, unsigned long index);
void GetRandomDouble(double &u, long input_key, long input_counter);

#endif
