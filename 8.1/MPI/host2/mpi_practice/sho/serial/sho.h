#ifndef sho_h
#define sho_h

// Paraemters for Simple Harmonic Oscillators

#define C_m 1.                  // mass
#define C_k 1.                  // spring constant
#define C_A 1.                  // amplitude
#define C_omega sqrt(C_k/C_m)   // angular frequency

enum _VARIABLES {phase, E, C_NUMBER_OF_VARIABLES};

#endif /* sho_h */
