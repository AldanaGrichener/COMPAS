#include "CircumbinaryDisk.h"

#include "constants.h"
#include "Options.h"
#include "utils.h"

#include <cmath>

namespace {

// File-local helpers for the Spindler-independent CBD formation criterion.
// These are deliberately kept out of the public CircumbinaryDisk namespace:
// the rest of COMPAS should see only the explicit input/result structs and
// EvaluateFormation(), not the implementation details of the available-AM
// estimate or the legacy synchronized-envelope comparison path.

constexpr double SYNCHRONIZED_ENVELOPE_INERTIA_COEFFICIENT = 0.2;

/*
 * Calculate the legacy synchronized-envelope estimate of the available angular momentum
 *
 * The estimate assumes the donor envelope is synchronized with the pre-RLOF
 * orbit and adds the envelope spin angular momentum to the pre-RLOF orbital
 * angular momentum.  The default formation path instead uses COMPAS's stored
 * total pre-RLOF binary angular momentum.
 *
 * double SynchronizedEnvelopeAngularMomentumEstimate(const CircumbinaryDisk::FormationInput& p_Input,
 *                                                     const double                             p_AvailableMassMsol)
 *
 * @param   [IN]    p_Input                    Physical state and model options for the formation criterion
 * @param   [IN]    p_AvailableMassMsol        Total binary mass lost during CE and available to the CBD model (Msol)
 * @return                                      Available angular momentum (Msol AU^2 yr^-1)
 */
double SynchronizedEnvelopeAngularMomentumEstimate(const CircumbinaryDisk::FormationInput& p_Input, const double p_AvailableMassMsol) {
    const double totalMassPreRLOF = p_Input.binaryMassPreCEMsol;
    const double reducedMassPreRLOF = p_Input.channel.reducedMassPreCEMsol;
    const double semiMajorAxisPreRLOFAU = p_Input.semiMajorAxisPreRLOFRsol * RSOL_TO_AU;
    const double envelopeDonorRadiusPreCEAU = p_Input.channel.envelopeDonorRadiusPreCERsol * RSOL_TO_AU;
    const double angularVelocityPreRLOF = std::sqrt(G_AU_Msol_yr * totalMassPreRLOF /
                                                    std::pow(semiMajorAxisPreRLOFAU, 3));
    const double envelopeMomentOfInertia = SYNCHRONIZED_ENVELOPE_INERTIA_COEFFICIENT *
                                           p_AvailableMassMsol *
                                           envelopeDonorRadiusPreCEAU *
                                           envelopeDonorRadiusPreCEAU;
    const double envelopeSpinAngularMomentum = angularVelocityPreRLOF * envelopeMomentOfInertia;
    const double orbitalAngularMomentumPreRLOF = reducedMassPreRLOF *
                                                 std::sqrt(G_AU_Msol_yr *
                                                           totalMassPreRLOF *
                                                           semiMajorAxisPreRLOFAU *
                                                           (1.0 - p_Input.eccentricityPreRLOF * p_Input.eccentricityPreRLOF));

    return orbitalAngularMomentumPreRLOF + envelopeSpinAngularMomentum;
}

}  // namespace

namespace CircumbinaryDisk {


/*
 * Copy the current global COMPAS options needed by the CBD formation model
 *
 * FormationOptions MakeFormationOptionsFromCompasOptions()
 *
 * @return                                      CBD formation options
 */
FormationOptions MakeFormationOptionsFromCompasOptions() {
    return {
        OPTIONS->CircumbinaryDiskUseSynchronizedEnvelopeAngularMomentum(),
        OPTIONS->CircumbinaryDiskBeta(),
        OPTIONS->CircumbinaryDiskEnvelopeMassFractionSupplied(),
        OPTIONS->CircumbinaryDiskStructureFactor(),
        OPTIONS->CircumbinaryDiskInnerRadiusOverSeparation()
    };
}

/*
 * Copy the current global COMPAS options needed by the CBD evolution model
 *
 * EvolutionOptions MakeEvolutionOptionsFromCompasOptions()
 *
 * @return                                      CBD evolution options
 */
EvolutionOptions MakeEvolutionOptionsFromCompasOptions() {
    return {
        OPTIONS->CircumbinaryDiskEddingtonMode(),
        OPTIONS->CircumbinaryDiskEddingtonCapFactor(),
        OPTIONS->CircumbinaryDiskEddingtonGe23Factor(),
        MCH
    };
}

/*
 * Determine whether a set of supernova-like events destroys an active CBD
 *
 * AIC is treated as a change to the live COMPAS stellar state, so a bound
 * system continues evolving through the same physical disk.
 *
 * bool SupernovaEventsDisruptDisk(const SN_EVENT p_Events)
 *
 * @param   [IN]    p_Events                   Supernova-event bitmask generated during the current event
 * @return                                      true for any non-AIC supernova event; false otherwise
 */
bool SupernovaEventsDisruptDisk(const SN_EVENT p_Events) {
    return p_Events != SN_EVENT::NONE && p_Events != SN_EVENT::AIC;
}

/*
 * Determine whether a CE has the supported compact-object plus envelope-donor geometry
 *
 * CommonEnvelopeChannel MakeCommonEnvelopeChannel(const STELLAR_TYPE p_StellarType1PreCE,
 *                                                 const STELLAR_TYPE p_StellarType2PreCE,
 *                                                 const double       p_Mass1PreCE,
 *                                                 const double       p_Mass2PreCE,
 *                                                 const double       p_Radius1PreCERsol,
 *                                                 const double       p_Radius2PreCERsol,
 *                                                 const bool         p_EnvelopeFlag1,
 *                                                 const bool         p_EnvelopeFlag2)
 *
 * @param   [IN]    p_StellarType1PreCE       Stellar type of star 1 immediately before CE
 * @param   [IN]    p_StellarType2PreCE       Stellar type of star 2 immediately before CE
 * @param   [IN]    p_Mass1PreCE              Mass of star 1 immediately before CE (Msol)
 * @param   [IN]    p_Mass2PreCE              Mass of star 2 immediately before CE (Msol)
 * @param   [IN]    p_Radius1PreCERsol        Radius of star 1 immediately before CE (Rsol)
 * @param   [IN]    p_Radius2PreCERsol        Radius of star 2 immediately before CE (Rsol)
 * @param   [IN]    p_EnvelopeFlag1           Whether star 1 has an envelope immediately before CE
 * @param   [IN]    p_EnvelopeFlag2           Whether star 2 has an envelope immediately before CE
 * @return                                      Current channel support, pre-CE reduced mass, and donor radius for the synchronized-envelope option
 */
CommonEnvelopeChannel MakeCommonEnvelopeChannel(const STELLAR_TYPE p_StellarType1PreCE,
                                                const STELLAR_TYPE p_StellarType2PreCE,
                                                const double       p_Mass1PreCE,
                                                const double       p_Mass2PreCE,
                                                const double       p_Radius1PreCERsol,
                                                const double       p_Radius2PreCERsol,
                                                const bool         p_EnvelopeFlag1,
                                                const bool         p_EnvelopeFlag2) {
    CommonEnvelopeChannel channel{};

    const STELLAR_TYPE_LIST supportedCompactObjects = {
        STELLAR_TYPE::HELIUM_WHITE_DWARF,
        STELLAR_TYPE::CARBON_OXYGEN_WHITE_DWARF,
        STELLAR_TYPE::OXYGEN_NEON_WHITE_DWARF,
        STELLAR_TYPE::NEUTRON_STAR,
        STELLAR_TYPE::BLACK_HOLE
    };
    const bool star1IsCompactObject = utils::IsOneOf(p_StellarType1PreCE, supportedCompactObjects);
    const bool star2IsCompactObject = utils::IsOneOf(p_StellarType2PreCE, supportedCompactObjects);

    if (star1IsCompactObject && p_EnvelopeFlag2 && !p_EnvelopeFlag1) {
        channel.envelopeDonorRadiusPreCERsol = p_Radius2PreCERsol;
    }
    else if (star2IsCompactObject && p_EnvelopeFlag1 && !p_EnvelopeFlag2) {
        channel.envelopeDonorRadiusPreCERsol = p_Radius1PreCERsol;
    }
    else {
        return channel;
    }

    const double binaryMassPreCEMsol = p_Mass1PreCE + p_Mass2PreCE;
    if (utils::Compare(binaryMassPreCEMsol, 0.0) <= 0) {
        return channel;
    }

    channel.supported = true;
    channel.reducedMassPreCEMsol = (p_Mass1PreCE * p_Mass2PreCE) / binaryMassPreCEMsol;
    return channel;
}

/*
 * Evaluate CBD formation criterion
 *
 * The mass supplied to the potential CBD is a fraction f_M of the decrease in total binary mass
 * across CE: (M1 + M2)_pre-CE - (M1 + M2)_post-CE. The resulting initial CBD
 * reservoir is assumed to satisfy m_CBD << M_binary during the evolution; formation does not
 * check this assumption.
 *
 * The criterion compares the total angular momentum assigned to the
 * CBD reservoir against the total angular momentum required for
 * supporting the CBD, given by the Keplerian AM at the adopted CBD inner radius times a disk structure factor.
 *
 * FormationResult EvaluateFormation(const FormationInput& p_Input)
 *
 * @param   [IN]    p_Input                    Physical state and model options for the formation criterion
 * @return                                      Formation outcome and supplied mass
 */
FormationResult EvaluateFormation(const FormationInput& p_Input) {
    FormationResult result{};

    const double availableMassMsol = p_Input.binaryMassPreCEMsol - p_Input.binaryMassPostCEMsol;
    if (utils::Compare(availableMassMsol, 0.0) <= 0 || utils::Compare(p_Input.semiMajorAxisPreRLOFRsol, 0.0) <= 0) {
        return result;
    }

    result.suppliedMassMsol = p_Input.options.envelopeMassFractionSupplied * availableMassMsol;
    if (utils::Compare(result.suppliedMassMsol, 0.0) <= 0) {
        return result;
    }

    const double availableAngularMomentum = p_Input.options.useSynchronizedEnvelopeAngularMomentum
                                                ? SynchronizedEnvelopeAngularMomentumEstimate(p_Input, availableMassMsol)
                                                : p_Input.totalAngularMomentumPreRLOF;
    // beta = fJ/fM, so the angular momentum assigned to the supplied CBD
    // reservoir is J_CBD = beta * fM * J_available.
    const double diskAngularMomentum = p_Input.options.circumbinaryDiskBeta *
                                       p_Input.options.envelopeMassFractionSupplied *
                                       availableAngularMomentum;

    // The adopted inner CBD radius scales with the post-CE separation because
    // the disk is assumed to orbit the post-CE binary, not the
    // larger pre-CE orbit.  Formation is allowed when the assigned angular
    // momentum meets or exceeds the threshold.
    const double innerDiskRadiusAU = p_Input.options.innerDiskRadiusOverSeparation * p_Input.semiMajorAxisPostCERsol * RSOL_TO_AU;
    const double thresholdAngularMomentum = result.suppliedMassMsol *
                                            p_Input.options.diskStructureAngularMomentumFactor *
                                            std::sqrt(G_AU_Msol_yr *
                                                      p_Input.binaryMassPostCEMsol *
                                                      innerDiskRadiusAU);

    result.forms = utils::Compare(diskAngularMomentum, thresholdAngularMomentum) >= 0;
    return result;
}

}  // namespace CircumbinaryDisk
