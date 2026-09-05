#ifndef __CircumbinaryDisk_h__
#define __CircumbinaryDisk_h__

#include "constants.h"
#include "typedefs.h"

namespace CircumbinaryDisk {

enum class StarLabel : int { NONE, STAR_1, STAR_2 };

// Numerical control for TIMESTEPPED mode: limit each active-CBD substep by
// the amount of supplied mass processed, since the Spindler derivatives are
// applied per unit supplied mass.  This is not a physical model parameter.
constexpr double MAXIMUM_SUPPLIED_MASS_PER_TIMESTEP_MSOL = 0.01;

// Identifies whether a pre-CE binary matches the currently supported
// compact-object + envelope-donor channel and stores the pre-CE quantities
// needed by the optional synchronized-envelope angular-momentum estimate.
struct CommonEnvelopeChannel {
    bool supported = false;
    double reducedMassPreCEMsol = 0.0;
    double envelopeDonorRadiusPreCERsol = 0.0;
};

// Public API for the post-CE circumbinary-disk model.
//
// BaseBinaryStar owns COMPAS state, event ordering, logging, and mutation of
// stars/binaries.  This module receives explicit physical inputs, evaluates the
// CBD formation/reservoir model, and returns CBD-driven binary changes without
// mutating COMPAS state.
//
// Validity domain: the Spindler-based evolution treats the CBD as a
// perturbative reservoir whose mass is dynamically negligible compared with
// the central binary.  The formalism therefore assumes m_CBD << M_binary.
// Here m_CBD is the initial intended reservoir (FormationResult::suppliedMassMsol
// and the logged Initial_CBD_Mass), M_binary is the post-CE central-binary mass. The code
// does not impose a hard numerical cutoff on this asymptotic assumption, so
// parameter choices outside this regime are extrapolations.
//
// Public structs use COMPAS star labels: star 1 and star 2 are never reordered
// at the module boundary.  Internally, the implementation handles the ordered
// q <= 1 convention required by the Siwek/Spindler tables and converts results
// back to COMPAS labels before returning them.
//
// Unit convention at the public boundary:
//   masses        : Msol
//   radii         : Rsol
//   separations   : Rsol, except semiMajorAxisFactor which is dimensionless
//   durations     : yr
//   eccentricity  : dimensionless

// User/model parameters for the formation criterion.  These are populated
// from COMPAS options by MakeFormationOptionsFromCompasOptions(); this struct
// intentionally carries no physical defaults.
struct FormationOptions {
    bool useSynchronizedEnvelopeAngularMomentum;
    double circumbinaryDiskBeta;
    double envelopeMassFractionSupplied;       // Assumed to yield initial m_CBD << post-CE M_binary
    double diskStructureAngularMomentumFactor;
    double innerDiskRadiusOverSeparation;
};

// User/model parameters for the CBD evolution step.  These are populated
// from COMPAS options by MakeEvolutionOptionsFromCompasOptions().
struct EvolutionOptions {
    CIRCUMBINARY_DISK_EDDINGTON_MODE eddingtonMode;
    double eddingtonCapFactor;
    double eddingtonGe23Factor;
    double aicMassThresholdMsol;
};

// Complete physical state needed for the CBD formation criterion.
// The caller identifies the compact-object companion and the envelope donor
// from the pre-CE COMPAS star labels and supplies their physical properties.
// beta is fJ/fM, where fJ is the fraction of the pre-CE binary angular momentum
// assigned to the CBD reservoir and fM is the supplied fraction of the total
// binary mass lost during CE.  Thus J_CBD = beta * fM * J_available.  The
// formation criterion is evaluated as a total angular-momentum comparison in
// COMPAS physical units. The available mass is calculated from the decrease
// in total binary mass across CE. The resulting supplied mass is assumed to 
// be much smaller than the post-CE binary mass in the subsequent Spindler evolution.
struct FormationInput {
    CommonEnvelopeChannel channel{};            // Pre-CE properties used by the synchronized-envelope option
    double binaryMassPreCEMsol;
    double binaryMassPostCEMsol;
    double semiMajorAxisPreRLOFRsol;
    double semiMajorAxisPostCERsol;
    double eccentricityPreRLOF;
    double totalAngularMomentumPreRLOF;
    FormationOptions options{};
};

// Outcome of the local CBD formation calculation, including the initial
// intended reservoir when the angular-momentum criterion is satisfied.
struct FormationResult {
    bool forms = false;
    double suppliedMassMsol = 0.0;              // Initial CBD reservoir; assumed to be << post-CE binary mass
};

// Complete physical state needed for one Spindler/CBD evolution step after a
// CBD has formed.  The supplied mass is the "bare" mass passed through the
// disk model; Eddington limiting determines how much of it is retained by each
// star. 
struct EvolutionInput {
    double mass1Msol;
    double mass2Msol;
    double radius1Rsol;
    double radius2Rsol;
    STELLAR_TYPE stellarType1;
    STELLAR_TYPE stellarType2;
    double eccentricity;
    double suppliedMassMsol;
    double durationYears;

    // Optional event target.  Use NONE to disable; otherwise use a COMPAS star label.
    // The current caller uses this to stop the CBD integration when an ONeWD
    // reaches the Chandrasekhar mass and should undergo AIC.
    StarLabel aicTargetStar;
    EvolutionOptions options{};
};

// COMPAS-labelled binary state and progress returned after evolving one
// requested CBD segment.  semiMajorAxisFactor multiplies the input separation;
// the remaining fields report the final orbit, masses, elapsed time, supplied
// mass processed, and whether evolution stopped at the requested AIC threshold.
struct EvolutionTrack {
    double semiMajorAxisFactor = 1.0;
    double eccentricity = 0.0;
    double mass1Msol = 0.0;
    double mass2Msol = 0.0;

    // If evolution stops at the requested AIC threshold, the orbital and mass
    // fields above already describe the state at the crossing.  These fields
    // report how much of the requested segment was completed.
    bool aicOccurred = false;
    double elapsedTimeYears = 0.0;
    double suppliedMassMsol = 0.0;
};

bool SupernovaEventsDisruptDisk(const SN_EVENT p_Events);

CommonEnvelopeChannel MakeCommonEnvelopeChannel(const STELLAR_TYPE p_StellarType1PreCE,
                                                const STELLAR_TYPE p_StellarType2PreCE,
                                                const double       p_Mass1PreCE,
                                                const double       p_Mass2PreCE,
                                                const double       p_Radius1PreCERsol,
                                                const double       p_Radius2PreCERsol,
                                                const bool         p_EnvelopeFlag1,
                                                const bool         p_EnvelopeFlag2);

FormationOptions MakeFormationOptionsFromCompasOptions();
EvolutionOptions MakeEvolutionOptionsFromCompasOptions();

FormationResult EvaluateFormation(const FormationInput& p_Input);

EvolutionTrack EvolveTrack(const EvolutionInput& p_Input);

}  // namespace CircumbinaryDisk

#endif  // __CircumbinaryDisk_h__
