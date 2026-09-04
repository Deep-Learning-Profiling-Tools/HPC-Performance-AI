/*
    Initializer:
    CBRNG_Random.cxx 
 
    File containing the function calls to generate random
    unsigned long integers (for the purpose of setting up
    keys) and doubles. Here we use the counter-based RNGs
    that are highly parallelizable.  

                    JD Emberson, March 2016
                       jemberson@anl.gov
 */

/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "CBRNG_Random.h"
#include "CBRNG_Uniform.h"

key_type GenerateKey(unsigned long input_key)
  {
    /* Generates the key array that is used to specify an independent family function of the CBRNG */
    
    key_type key={{0}};

    key[0] = input_key;

    return key;

  }

ctr_type GenerateCounter(unsigned long input_counter)
  {
    /* The counter sets the index in the family function identified by the key. A unique counter for a
       given key will always generate the same random number */

    ctr_type cindex={{0}};

    cindex[0] = 1664525*input_counter + 1013904223;
    cindex[1] = 7583981*input_counter + 1128198847;

    return cindex;

  }

ctr_type GetRandomUnsignedLong(unsigned long input_key, unsigned long input_counter)
  {
    /* Returns the unique unsigned long integer for a family function specified by input_key
       and counter specified by input_counter. */    

    RNG rng;

    // Initialize the key and counter
    key_type key    = GenerateKey(input_key);
    ctr_type cindex = GenerateCounter(input_counter);

    // Return long integers
    ctr_type r = rng(cindex, key);

    return r;

  }

void GetSlabKeys(unsigned long *keys, int x1, int numx, unsigned long seed)
  {
    /* Random numbers are generated on the 3D grid by having a unique key assigned
       to each slab of the grid along the x axis. These slab keys are generated
       from the user-supplied global seed in the params file. */

    unsigned long counter;
    
    for (int i=0; i<numx; i++)
      {
        counter = x1 + i;
        ctr_type r = GetRandomUnsignedLong(seed, counter); 
        keys[i] = r[0];
      } 

  }

void GetRandomDoublesWhiteNoise(double &rn1, double &rn2, unsigned long input_key, unsigned long input_counter)
  {
    /* Returns two random numbers in rn1 and rn2 for the construction of a white noise field in real space */

    RNG rng;
    ctr_type r = {{0}};
    double rn;

    // Initialize the key and counter
    key_type key    = GenerateKey(input_key);
    ctr_type cindex = GenerateCounter(input_counter);

    do 
    {
        r = rng(cindex, key);
        rn1 = -1 + 2*r123::u01<double>(r[0]);
        rn2 = -1 + 2*r123::u01<double>(r[1]);
        rn = rn1*rn1 + rn2*rn2;
        cindex.incr();        
    } while (rn > 1.0 || rn == 0);
 
  }
  
void GetRandomDouble(double &u, long input_key, long input_counter)
  {
    /* Draws random doubles from the family funcion with input_key and index input_counter.
       Uses the u01<double> routine specified in CBRNG_Uniform.h to convert form ulong's to doubles. */ 

    ctr_type r = GetRandomUnsignedLong(input_key, input_counter);
//    for (int i=0; i<nsize_r123; i++) u[i] = r123::u01<double>(r[i]);
    u = r123::u01<double>(r[0]);
  }

