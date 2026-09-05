#include "CircumbinaryDisk.h"

#include "constants.h"
#include "utils.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <boost/numeric/odeint.hpp>

// spindler-c is a C library. Its public header currently does not wrap the
// declarations in extern "C", so we do it here for C++/COMPAS builds.
extern "C" {
#include "spindler.h"
}

namespace odeint = boost::numeric::odeint;

// Physics scope of this module:
//   * Disk formation is handled in CircumbinaryDiskFormation.cpp.  This file
//     assumes a CBD reservoir has already formed and evolves the binary using
//     the Siwek/Spindler tabulated derivatives for circumbinary-disk interaction.
//   * The supplied reservoir is delivered at a constant rate over the chosen
//     disk lifetime.  Each component accretes the table-implied fraction unless
//     limited by the selected Eddington prescription; unretained mass removes
//     angular momentum from the binary using the accretor's specific angular momentum.

namespace {

enum class OrderedComponent { NONE, PRIMARY, SECONDARY };

// Physical and dimensionless inputs used by the internal Spindler integration.
// suppliedMassFraction is the total mass supplied by the CBD in units of the
// initial post-CE binary mass and is assumed to be much smaller than unity; a
// small TIMESTEPPED increment does not by itself make an initially large
// reservoir consistent with that assumption. q0 follows the ordered
// Spindler convention q = M2/M1 <= 1, not necessarily the original COMPAS
// star-2/star-1 label ratio. initialTotalMass converts between physical Msol
// units and the dimensionless ODE variables.
struct SpindlerEvolutionParams {
    double suppliedMassFraction;
    double q0;
    double e0;
    double initialTotalMass;
    double durationYears;

    STELLAR_TYPE stellarType1 = STELLAR_TYPE::NONE;
    STELLAR_TYPE stellarType2 = STELLAR_TYPE::NONE;
    double radius1 = 0.0;
    double radius2 = 0.0;

    CircumbinaryDisk::EvolutionOptions options{};

    // Optional stop condition for event-style AIC handling.  The component
    // label is in the ordered Spindler convention; the threshold comes from
    // options.aicMassThresholdMsol.
    OrderedComponent stopComponent;
};

// Final dimensionless state and integration progress produced by the internal
// Spindler solver before component ordering is converted back to COMPAS labels.
struct SpindlerEvolutionResult {
    double semiMajorAxisFactor = 1.0;
    double mass1 = 0.5;
    double mass2 = 0.5;
    double eccentricity = 0.0;
    double elapsedTimeYears = 0.0;
    double suppliedMassFractionUsed = 0.0;
    bool stoppedAtMassThreshold = false;
};

// Owns one initialized spindler-c table set and evaluates CBD-driven binary
// evolution with COMPAS Eddington-limit and angular-momentum-loss extensions.
class SpindlerBackend {
public:
    explicit SpindlerBackend(const std::string& p_ModelName = "Siwek23");
    ~SpindlerBackend();

    // The backend uniquely owns spindler-c table data, so copying and moving are disabled.
    SpindlerBackend(const SpindlerBackend&) = delete;
    SpindlerBackend& operator=(const SpindlerBackend&) = delete;
    SpindlerBackend(SpindlerBackend&&) = delete;
    SpindlerBackend& operator=(SpindlerBackend&&) = delete;
    // Evolve with Eddington-limited, non-conservative accretion.
    SpindlerEvolutionResult Evolve(const SpindlerEvolutionParams& p_Params);

private:
    // Spindler logarithmic orbital and mass-ratio derivatives returned for one
    // labelled binary state.
    struct Derivatives {
        double Da;
        double Dq;
        double De;
    };

    Derivatives DerivativesForLabeledQ(double p_QLabeled, double p_Eccentricity) const;
    std::pair<double, double> ComponentMassFractions(double p_QLabeled, double p_Eccentricity) const;
    double ComputeEddingtonRate(
        const STELLAR_TYPE p_StellarType,
        double p_MassSpindlerUnits,
        double p_InitialTotalMass,
        double p_Radius
    ) const;

    static double ApplyEddingtonLimit(
        double p_RequestedRate,
        double p_EddingtonRate,
        CIRCUMBINARY_DISK_EDDINGTON_MODE p_Mode,
        double p_CapFactor,
        double p_Ge23Factor
    );

    // Dimensionless Spindler-internal orbital angular momentum: G = 1, a0 = 1, and Mtot0 = 1.
    static double SpindlerOrbitalAngularMomentum(double p_Mass1, double p_Mass2, double p_Eccentricity, double p_SemiMajorAxis);

    struct spindler_data_t m_SpindlerData{};
};

#ifndef SPINDLER_C_ROOT
#define SPINDLER_C_ROOT ""
#endif

// Range of the Siwek23/Spindler tables used by spindler-c.  Systems outside
// the table domain are evolved using the nearest tabulated boundary value,
// while the raw q/e values are retained in the derivative normalisations so the
// evolution remains continuous at the table edges.
constexpr double SIWEK23_Q_MIN = 0.1;
constexpr double SIWEK23_Q_MAX = 1.0;
constexpr double SIWEK23_E_MIN = 0.0;
constexpr double SIWEK23_E_MAX = 0.8;
constexpr double NS_REFERENCE_RADIUS_RSUN = 1.4374e-5;
constexpr double WD_EDDINGTON_RATE_PER_RADIUS = 2.08e-3 / 1.7;
constexpr double SPINDLER_ODE_RTOL = 1.0e-4;
constexpr double SPINDLER_ODE_ATOL = 1.0e-5;
constexpr double TINY = std::numeric_limits<double>::min();

/*
 * Validate the ordered initial mass ratio and eccentricity before constructing the ODE state
 *
 * void ValidateCommonInputs(const double p_Q0, const double p_E0)
 *
 * @param   [IN]    p_Q0                       Initial ordered mass ratio, M2/M1 <= 1
 * @param   [IN]    p_E0                       Initial eccentricity
 */
void ValidateCommonInputs(const double p_Q0, const double p_E0) {
    if (utils::Compare(p_Q0, 0.0) <= 0 || !std::isfinite(p_Q0)) {
        throw std::invalid_argument("q0 must be positive and finite");
    }
    if (!std::isfinite(p_E0) || utils::Compare(p_E0, 0.0) < 0 || utils::Compare(p_E0, 1.0) >= 0) {
        throw std::invalid_argument("e0 must satisfy 0 <= e0 < 1");
    }
}

/*
 * Convert the dimensionless ODE state into the internal Spindler result
 *
 * SpindlerEvolutionResult MakeResultFromState(const std::vector<double>& p_State,
 *                                               const double               p_ElapsedTimeYears,
 *                                               const double               p_DurationYears,
 *                                               const double               p_SuppliedMassFraction,
 *                                               const bool                 p_StoppedAtMassThreshold)
 *
 * @param   [IN]    p_State                    ODE state [a/a0, M1/M0, M2/M0, e]
 * @param   [IN]    p_ElapsedTimeYears         Time evolved from the start of this CBD segment (yr)
 * @param   [IN]    p_DurationYears            Requested duration of this CBD segment (yr)
 * @param   [IN]    p_SuppliedMassFraction     Total supplied mass divided by the initial binary mass
 * @param   [IN]    p_StoppedAtMassThreshold   Whether integration stopped at the requested mass threshold
 * @return                                      Internal evolution result at p_ElapsedTimeYears
 */
SpindlerEvolutionResult MakeResultFromState(
    const std::vector<double>& p_State,
    const double p_ElapsedTimeYears,
    const double p_DurationYears,
    const double p_SuppliedMassFraction,
    const bool p_StoppedAtMassThreshold
) {
    SpindlerEvolutionResult result{};
    result.semiMajorAxisFactor = p_State[0];
    result.mass1 = p_State[1];
    result.mass2 = p_State[2];
    result.eccentricity = p_State[3];
    result.elapsedTimeYears = p_ElapsedTimeYears;
    result.suppliedMassFractionUsed = p_SuppliedMassFraction * p_ElapsedTimeYears / p_DurationYears;
    result.stoppedAtMassThreshold = p_StoppedAtMassThreshold;
    return result;
}

/*
 * Read the selected ordered-component mass from the ODE state
 *
 * double SelectedComponentMass(const std::vector<double>& p_State,
 *                              const OrderedComponent     p_Component)
 *
 * @param   [IN]    p_State                    ODE state [a/a0, M1/M0, M2/M0, e]
 * @param   [IN]    p_Component                Ordered component to read, or NONE when no event target is active
 * @return                                      Selected dimensionless component mass; zero for NONE
 */
double SelectedComponentMass(const std::vector<double>& p_State, const OrderedComponent p_Component) {
    switch (p_Component) {
        case OrderedComponent::PRIMARY:   return p_State[1];
        case OrderedComponent::SECONDARY: return p_State[2];
        case OrderedComponent::NONE:      return 0.0;
    }

    return 0.0;
}

std::mutex spindlerInitMutex;

// spindler_init() resolves its table paths relative to the process working
// directory. This RAII guard temporarily switches to the spindler-c root and
// guarantees that COMPAS's original working directory is restored on exit.
class ScopedWorkingDirectory {
public:
    /*
     * Temporarily change the process working directory
     *
     * explicit ScopedWorkingDirectory(const std::string& p_NewDirectory)
     *
     * @param   [IN]    p_NewDirectory             Directory required while spindler_init() resolves table paths
     */
    explicit ScopedWorkingDirectory(const std::string& p_NewDirectory) {
        if (p_NewDirectory.empty()) return;

        char* cwd = getcwd(nullptr, 0);
        if (cwd == nullptr) {
            std::ostringstream msg;
            msg << "getcwd failed before spindler_init: " << std::strerror(errno);
            throw std::runtime_error(msg.str());
        }
        m_OldDirectory = cwd;
        std::free(cwd);

        if (chdir(p_NewDirectory.c_str()) != 0) {
            std::ostringstream msg;
            msg << "Could not chdir to SPINDLER_C_ROOT='" << p_NewDirectory
                << "' before spindler_init: " << std::strerror(errno);
            throw std::runtime_error(msg.str());
        }
        m_Changed = true;
    }

    /*
     * Restore the process working directory saved by the constructor
     *
     * ~ScopedWorkingDirectory()
     */
    ~ScopedWorkingDirectory() {
        if (m_Changed) {
            (void)chdir(m_OldDirectory.c_str());
        }
    }

    // Keep directory-restoration ownership unique to a single scope guard.
    ScopedWorkingDirectory(const ScopedWorkingDirectory&) = delete;
    ScopedWorkingDirectory& operator=(const ScopedWorkingDirectory&) = delete;

private:
    std::string m_OldDirectory;
    bool m_Changed = false;
};

/*
 * Initialise one spindler-c table backend
 *
 * SpindlerBackend(const std::string& p_ModelName)
 *
 * @param   [IN]    p_ModelName                Name of the spindler-c table model to load
 */
SpindlerBackend::SpindlerBackend(const std::string& p_ModelName) {
    std::vector<char> modelNameCString(p_ModelName.begin(), p_ModelName.end());
    modelNameCString.push_back('\0');

    // If enabled, the Makefile defines SPINDLER_C_ROOT to the user-supplied
    // spindler-c root.  spindler-c currently looks for tables/<model>/...
    // relative to the current working directory during spindler_init(), so we
    // temporarily initialize from that directory and then restore COMPAS's
    // original working directory.

    std::lock_guard<std::mutex> initLock(spindlerInitMutex);
    ScopedWorkingDirectory cwdGuard{SPINDLER_C_ROOT};

    const int err = spindler_init(modelNameCString.data(), &m_SpindlerData);
    if (err != SPINDLER_NO_ERROR) {
        std::ostringstream msg;
        msg << "spindler_init failed for model '" << p_ModelName
            << "' with error code " << err
            << ". Check SPINDLER_C_ROOT='" << SPINDLER_C_ROOT
            << "' and verify that tables/" << p_ModelName << " exists there.";
        throw std::runtime_error(msg.str());
    }
}

/*
 * Release the table data allocated by spindler_init()
 *
 * ~SpindlerBackend()
 */
SpindlerBackend::~SpindlerBackend() {
    spindler_free_data(&m_SpindlerData);
}

/*
 * Query the tabulated logarithmic orbital derivatives
 *
 * Derivatives DerivativesForLabeledQ(double p_QLabeled, double p_Eccentricity) const
 *
 * @param   [IN]    p_QLabeled                 Mass ratio in the caller's current component labels
 * @param   [IN]    p_Eccentricity             Current eccentricity
 * @return                                      Da, Dq, and De mapped back to the caller's labels
 */
SpindlerBackend::Derivatives SpindlerBackend::DerivativesForLabeledQ(double p_QLabeled, double p_Eccentricity) const {
    if (utils::Compare(p_QLabeled, 0.0) <= 0 || !std::isfinite(p_QLabeled)) {
        throw std::runtime_error("Encountered non-positive or non-finite mass ratio during Spindler integration");
    }

    const bool swapped = utils::Compare(p_QLabeled, 1.0) > 0;
    const double qOrderedRaw = swapped ? 1.0 / p_QLabeled : p_QLabeled;
    const double eccentricityRaw = std::max(p_Eccentricity, 0.0);

    // The table coefficients are logarithmic responses of the binary to CBD
    // accretion/torques.  Da multiplies a, De multiplies e, and Dq describes the
    // evolution of the ordered mass ratio.  If COMPAS's labelled mass ratio is
    // inverted relative to q <= 1, the sign of the q derivative must be inverted
    // before returning to the caller's labels.
    const double qLookup = std::clamp(qOrderedRaw, SIWEK23_Q_MIN, SIWEK23_Q_MAX);
    const double eccentricityLookup = std::clamp(eccentricityRaw, SIWEK23_E_MIN, SIWEK23_E_MAX);

    Derivatives d{};
    d.Da = spindler_get_Da(qLookup, eccentricityLookup, const_cast<struct spindler_data_t*>(&m_SpindlerData));

    const double DeLookup = spindler_get_De(qLookup, eccentricityLookup, const_cast<struct spindler_data_t*>(&m_SpindlerData));
    d.De = utils::Compare(eccentricityRaw, 0.0) > 0 ? DeLookup * eccentricityLookup / eccentricityRaw : 0.0;

    const double DqLookup = spindler_get_Dq(qLookup, eccentricityLookup, const_cast<struct spindler_data_t*>(&m_SpindlerData));
    const double DqOrdered = DqLookup * qLookup / qOrderedRaw;
    d.Dq = swapped ? -DqOrdered : DqOrdered;
    return d;
}

/*
 * Infer the component shares of the supplied CBD mass
 *
 * std::pair<double, double> ComponentMassFractions(double p_QLabeled, double p_Eccentricity) const
 *
 * @param   [IN]    p_QLabeled                 Mass ratio in the caller's current component labels
 * @param   [IN]    p_Eccentricity             Current eccentricity
 * @return                                      Fractions of supplied mass assigned to components 1 and 2
 */
std::pair<double, double> SpindlerBackend::ComponentMassFractions(double p_QLabeled, double p_Eccentricity) const {
    if (utils::Compare(p_QLabeled, 0.0) <= 0 || !std::isfinite(p_QLabeled)) {
        throw std::runtime_error("Encountered non-positive or non-finite mass ratio while computing accretion split");
    }

    const bool swapped = utils::Compare(p_QLabeled, 1.0) > 0;
    const double qOrderedRaw = swapped ? 1.0 / p_QLabeled : p_QLabeled;
    const double eccentricityRaw = std::max(p_Eccentricity, 0.0);
    const double qLookup = std::clamp(qOrderedRaw, SIWEK23_Q_MIN, SIWEK23_Q_MAX);
    const double eccentricityLookup = std::clamp(eccentricityRaw, SIWEK23_E_MIN, SIWEK23_E_MAX);

    const double DqLookup = spindler_get_Dq(qLookup, eccentricityLookup, const_cast<struct spindler_data_t*>(&m_SpindlerData));
    const double DqOrdered = DqLookup * qLookup / qOrderedRaw;

    // Infer how the supplied CBD mass is split between the two components from
    // the tabulated q evolution.  This keeps the mass growth consistent with the
    // same table derivative that drives the orbital response instead of imposing
    // an independent accretion-ratio prescription.
    double fracSecondaryLookup = qOrderedRaw * (DqOrdered + 1.0 + qOrderedRaw) / std::pow(1.0 + qOrderedRaw, 2.0);
    fracSecondaryLookup = std::clamp(fracSecondaryLookup, 0.0, 1.0);
    const double fracPrimaryLookup = 1.0 - fracSecondaryLookup;

    if (swapped) {
        return {fracSecondaryLookup, fracPrimaryLookup};
    }
    return {fracPrimaryLookup, fracSecondaryLookup};
}

/*
 * Calculate the Eddington accretion rate for one component
 *
 * double ComputeEddingtonRate(const STELLAR_TYPE p_StellarType,
 *                             double             p_MassSpindlerUnits,
 *                             double             p_InitialTotalMass,
 *                             double             p_Radius) const
 *
 * @param   [IN]    p_StellarType             Current stellar type of the component
 * @param   [IN]    p_MassSpindlerUnits       Component mass divided by the initial binary mass
 * @param   [IN]    p_InitialTotalMass         Initial binary mass used for ODE normalisation (Msol)
 * @param   [IN]    p_Radius                   Current component radius (Rsol)
 * @return                                      Eddington rate in dimensionless Spindler mass units per year
 */
double SpindlerBackend::ComputeEddingtonRate(
    const STELLAR_TYPE p_StellarType,
    double p_MassSpindlerUnits,
    double p_InitialTotalMass,
    double p_Radius
) const {
    if (p_StellarType == STELLAR_TYPE::BLACK_HOLE) {
        //  Black-hole Eddington mass accretion rate = 4.42e-7 * (M_phys / 10) Msol/yr.  Because the ODE masses
        // are normalised by the initial binary mass, this is returned in
        // dimensionless Spindler mass units per year.
        return 4.42e-7 * (p_MassSpindlerUnits / 10.0);
    }

    const bool isWhiteDwarf = p_StellarType == STELLAR_TYPE::HELIUM_WHITE_DWARF ||
                              p_StellarType == STELLAR_TYPE::CARBON_OXYGEN_WHITE_DWARF ||
                              p_StellarType == STELLAR_TYPE::OXYGEN_NEON_WHITE_DWARF;

    if (p_StellarType == STELLAR_TYPE::NEUTRON_STAR || isWhiteDwarf) {
        if (utils::Compare(p_Radius, 0.0) <= 0 || !std::isfinite(p_Radius)) {
            throw std::invalid_argument("radius must be positive and finite for radius-scaled Eddington limits");
        }
        if (utils::Compare(p_InitialTotalMass, 0.0) <= 0 || !std::isfinite(p_InitialTotalMass)) {
            throw std::invalid_argument("initialTotalMass must be positive for radius-scaled Eddington limits");
        }

        // COMPAS uses a radius-scaled Eddington rate for white dwarfs and other
        // non-BH accretors. Preserve the existing CBD neutron-star normalisation
        // while applying the standard COMPAS/Hurley coefficient to white dwarfs.
        const double mdotEddPhys = p_StellarType == STELLAR_TYPE::NEUTRON_STAR
                                     ? 3.0e-8 * (p_Radius / NS_REFERENCE_RADIUS_RSUN)
                                     : WD_EDDINGTON_RATE_PER_RADIUS * p_Radius;
        return mdotEddPhys / p_InitialTotalMass;
    }

    // The Eddington limit is assumed to be irrelevant for non-compact accretors.
    return std::numeric_limits<double>::infinity();
}

/*
 * Apply the selected Eddington-limited retention prescription
 *
 * CAP imposes a hard multiple of the Eddington rate, TM23 reduces the
 * super-Eddington rate following Tuna & Metzger (2023), and GE23 retains a
 * fixed fraction of the requested super-Eddington rate.
 *
 * double ApplyEddingtonLimit(double                                p_RequestedRate,
 *                            double                                p_EddingtonRate,
 *                            CIRCUMBINARY_DISK_EDDINGTON_MODE      p_Mode,
 *                            double                                p_CapFactor,
 *                            double                                p_Ge23Factor)
 *
 * @param   [IN]    p_RequestedRate            Accretion rate requested by the Spindler mass split
 * @param   [IN]    p_EddingtonRate            Eddington accretion rate in matching units
 * @param   [IN]    p_Mode                      Eddington-limiting prescription
 * @param   [IN]    p_CapFactor                 Multiple of the Eddington rate allowed in CAP mode
 * @param   [IN]    p_Ge23Factor                Retained fraction above Eddington in GE23 mode
 * @return                                      Retained accretion rate
 */
double SpindlerBackend::ApplyEddingtonLimit(
    double p_RequestedRate,
    double p_EddingtonRate,
    CIRCUMBINARY_DISK_EDDINGTON_MODE p_Mode,
    double p_CapFactor,
    double p_Ge23Factor
) {
    switch (p_Mode) {
        case CIRCUMBINARY_DISK_EDDINGTON_MODE::CAP: {
            return std::min(p_RequestedRate, p_CapFactor * p_EddingtonRate);
        }

        case CIRCUMBINARY_DISK_EDDINGTON_MODE::TM23: {
            if (std::isfinite(p_EddingtonRate) && utils::Compare(p_EddingtonRate, 0.0) > 0 && utils::Compare(p_RequestedRate, p_EddingtonRate) > 0) {
                const double ratio = p_RequestedRate / p_EddingtonRate;
                const double fTM23 = std::min(1.0, std::pow(0.6 / ratio, 0.6));
                return p_RequestedRate * fTM23;
            }
            return p_RequestedRate;
        }

        case CIRCUMBINARY_DISK_EDDINGTON_MODE::GE23: {
            return utils::Compare(p_RequestedRate, p_EddingtonRate) > 0 ? p_RequestedRate * p_Ge23Factor : p_RequestedRate;
        }
    }

    throw std::invalid_argument("eddingtonMode must be 'cap', 'ge23', or 'tm23'");
}

/*
 * Calculate dimensionless orbital angular momentum in the Spindler normalisation
 *
 * double SpindlerOrbitalAngularMomentum(double p_Mass1,
 *                                       double p_Mass2,
 *                                       double p_Eccentricity,
 *                                       double p_SemiMajorAxis)
 *
 * @param   [IN]    p_Mass1                    Ordered primary mass in Spindler units
 * @param   [IN]    p_Mass2                    Ordered secondary mass in Spindler units
 * @param   [IN]    p_Eccentricity             Current eccentricity
 * @param   [IN]    p_SemiMajorAxis            Current semi-major axis divided by its initial value
 * @return                                      Orbital angular momentum for G = Mtot,0 = a0 = 1
 */
double SpindlerBackend::SpindlerOrbitalAngularMomentum(double p_Mass1, double p_Mass2, double p_Eccentricity, double p_SemiMajorAxis) {
    const double totalMass = p_Mass1 + p_Mass2;
    const double reducedMass = p_Mass1 * p_Mass2 / totalMass;
    const double oneMinusE2 = std::max(1.0 - p_Eccentricity * p_Eccentricity, TINY);
    return reducedMass * std::sqrt(totalMass * p_SemiMajorAxis * oneMinusE2);
}

/*
 * Evolve the Spindler system with Eddington-limited accretion
 *
 * State vector: [a/a0, M1/M0, M2/M0, e] in the ordered Spindler labels.  The
 * independent variable is time in years.  The supplied CBD mass is spread
 * uniformly over p_Params.durationYears, then split between the two stars using
 * the Spindler qdot-derived accretion fractions. The encompassing CBD model assumes the initial
 * reservoir, Initial_CBD_Mass, satisfies m_CBD << M_binary.
 *
 * The orbital terms have two pieces:
 *   - CBD torque/accretion response from the Spindler Da and De tables.
 *   - Extra non-conservative widening/shrinking from angular momentum carried
 *     away by material that was supplied but ejected due to Eddington limits.
 *
 * The table torques scale with the supplied mass flux.  Eccentricity is evolved
 * only through the tabulated CBD response; the simple wind-loss term affects a
 * through Jdot but does not add an independent de/dt prescription.
 *
 * SpindlerEvolutionResult Evolve(const SpindlerEvolutionParams& p_Params)
 *
 * @param   [IN]    p_Params                   Ordered, dimensionless initial state and model parameters
 * @return                                      State reached at the requested duration or mass-threshold crossing
 */
SpindlerEvolutionResult SpindlerBackend::Evolve(const SpindlerEvolutionParams& p_Params) {
    ValidateCommonInputs(p_Params.q0, p_Params.e0);

    if (utils::Compare(p_Params.suppliedMassFraction, 0.0) < 0 || !std::isfinite(p_Params.suppliedMassFraction)) {
        throw std::invalid_argument("suppliedMassFraction must be non-negative and finite");
    }
    if (utils::Compare(p_Params.initialTotalMass, 0.0) <= 0 || !std::isfinite(p_Params.initialTotalMass)) {
        throw std::invalid_argument("initialTotalMass must be positive and finite");
    }
    if (utils::Compare(p_Params.durationYears, 0.0) <= 0 || !std::isfinite(p_Params.durationYears)) {
        throw std::invalid_argument("duration must be positive and finite");
    }
    if (utils::Compare(p_Params.options.eddingtonCapFactor, 0.0) <= 0 || !std::isfinite(p_Params.options.eddingtonCapFactor)) {
        throw std::invalid_argument("eddingtonCapFactor must be positive and finite");
    }
    if (!(utils::Compare(p_Params.options.eddingtonGe23Factor, 0.0) > 0 && utils::Compare(p_Params.options.eddingtonGe23Factor, 1.0) <= 0) ||
        !std::isfinite(p_Params.options.eddingtonGe23Factor)) {
        throw std::invalid_argument("eddingtonGe23Factor must satisfy 0 < f <= 1");
    }

    if (p_Params.stopComponent != OrderedComponent::NONE && utils::Compare(p_Params.options.aicMassThresholdMsol, 0.0) <= 0) {
        throw std::invalid_argument("aicMassThresholdMsol must be positive when stopComponent is set");
    }

    const double stopMassThreshold = p_Params.stopComponent == OrderedComponent::NONE
                                         ? 0.0
                                         : p_Params.options.aicMassThresholdMsol / p_Params.initialTotalMass;

    using StateType = std::vector<double>;
    StateType state = {1.0, 1.0 / (1.0 + p_Params.q0), p_Params.q0 / (1.0 + p_Params.q0), p_Params.e0};

    if (utils::Compare(p_Params.suppliedMassFraction, 0.0) == 0) {
        return MakeResultFromState(state, 0.0, p_Params.durationYears, p_Params.suppliedMassFraction, false);
    }

    const double mdotSupply = p_Params.suppliedMassFraction / p_Params.durationYears;

    // Define the ODE derivatives for orbital size, component masses, and
    // eccentricity under CBD torques plus non-conservative mass loss.
    auto system = [this, &p_Params, mdotSupply](const StateType& p_XState, StateType& p_DxDt, const double /*p_Time*/) {
        p_DxDt.resize(4);
        const double aState = p_XState[0];
        const double mass1 = p_XState[1];
        const double mass2 = p_XState[2];
        const double eState = p_XState[3];

        if (utils::Compare(aState, 0.0) <= 0 || utils::Compare(mass1, 0.0) <= 0 || utils::Compare(mass2, 0.0) <= 0) {
            throw std::runtime_error("Non-positive a, M1, or M2 encountered during Spindler integration");
        }

        const double totalMass = mass1 + mass2;
        const double qState = mass2 / mass1;
        const double orbitalAngularMomentum = SpindlerOrbitalAngularMomentum(mass1, mass2, eState, aState);

        const double mdotEdd1 = ComputeEddingtonRate(
            p_Params.stellarType1, mass1, p_Params.initialTotalMass, p_Params.radius1
        );
        const double mdotEdd2 = ComputeEddingtonRate(
            p_Params.stellarType2, mass2, p_Params.initialTotalMass, p_Params.radius2
        );

        const auto fractions = ComponentMassFractions(qState, eState);
        const double mdotReq1 = fractions.first * mdotSupply;
        const double mdotReq2 = fractions.second * mdotSupply;

        const double mdotAcc1 = ApplyEddingtonLimit(
            mdotReq1,
            mdotEdd1,
            p_Params.options.eddingtonMode,
            p_Params.options.eddingtonCapFactor,
            p_Params.options.eddingtonGe23Factor
        );
        const double mdotAcc2 = ApplyEddingtonLimit(
            mdotReq2,
            mdotEdd2,
            p_Params.options.eddingtonMode,
            p_Params.options.eddingtonCapFactor,
            p_Params.options.eddingtonGe23Factor
        );
        const double mdotLost1 = std::max(mdotReq1 - mdotAcc1, 0.0);
        const double mdotLost2 = std::max(mdotReq2 - mdotAcc2, 0.0);

        // The Spindler tables give the CBD-driven response per unit supplied mass
        // through the disk/binary interaction.
        const auto d = DerivativesForLabeledQ(qState, eState);
        const double dadtDisk = d.Da * aState * mdotSupply / totalMass;
        const double dedt = d.De * eState * mdotSupply / totalMass;

        const double jBinary = orbitalAngularMomentum / totalMass;
        const double j1 = qState * jBinary;
        const double j2 = jBinary / qState;
        const double dJdtWind = -(j1 * mdotLost1 + j2 * mdotLost2);
        const double dadtWind = utils::Compare(orbitalAngularMomentum, 0.0) > 0 ? 2.0 * aState * dJdtWind / orbitalAngularMomentum : 0.0;

        p_DxDt[0] = dadtDisk + dadtWind;
        p_DxDt[1] = mdotAcc1;
        p_DxDt[2] = mdotAcc2;
        p_DxDt[3] = dedt;
    };

    auto stepper = odeint::make_controlled(
        SPINDLER_ODE_ATOL,
        SPINDLER_ODE_RTOL,
        odeint::runge_kutta_cash_karp54<StateType>()
    );

    double time = 0.0;
    double dt = std::max(p_Params.durationYears / 1000.0, 1.0e-8);

    const bool stopOnMassThreshold = p_Params.stopComponent != OrderedComponent::NONE;
    const double initialTargetMass = SelectedComponentMass(state, p_Params.stopComponent);
    if (stopOnMassThreshold && utils::Compare(initialTargetMass, stopMassThreshold) >= 0) {
        return MakeResultFromState(state, 0.0, p_Params.durationYears, p_Params.suppliedMassFraction, true);
    }

    // Deliberately use strict comparisons here rather than utils::Compare():
    // the ODE controller must advance to, and clip exactly at, the requested
    // endpoint instead of treating a nearby time as equivalent.
    while (time < p_Params.durationYears) {
        if (time + dt > p_Params.durationYears) {
            dt = p_Params.durationYears - time;
        }

        const StateType stateBeforeStep = state;
        const double timeBeforeStep = time;
        const double targetBeforeStep = SelectedComponentMass(stateBeforeStep, p_Params.stopComponent);

        boost::numeric::odeint::controlled_step_result stepResult;
        do {
            stepResult = stepper.try_step(system, state, time, dt);
        } while (stepResult == boost::numeric::odeint::fail);

        if (stopOnMassThreshold) {
            const double targetAfterStep = SelectedComponentMass(state, p_Params.stopComponent);
            // Deliberately use strict comparisons here rather than utils::Compare():
            // this is a sign-change bracket for interpolation to the AIC mass
            // crossing; tolerance-based equality can erase the bracket.
            if (targetBeforeStep < stopMassThreshold && targetAfterStep >= stopMassThreshold) {
                // Event-style AIC handling: do a refined integration to the
                // mass-threshold crossing and return that intermediate state to
                // COMPAS, rather than evolving past the stellar-type change.
                const double fraction = (stopMassThreshold - targetBeforeStep) / (targetAfterStep - targetBeforeStep);
                const double crossingTime = timeBeforeStep + fraction * (time - timeBeforeStep);
                StateType crossingState = stateBeforeStep;
                double crossingIntegrationTime = timeBeforeStep;
                double crossingDt = std::max((crossingTime - timeBeforeStep) / 10.0, 1.0e-10);
                auto crossingStepper = odeint::make_controlled(
                    SPINDLER_ODE_ATOL,
                    SPINDLER_ODE_RTOL,
                    odeint::runge_kutta_cash_karp54<StateType>()
                );
                odeint::integrate_adaptive(
                    crossingStepper,
                    system,
                    crossingState,
                    crossingIntegrationTime,
                    crossingTime,
                    crossingDt
                );
                return MakeResultFromState(
                    crossingState,
                    crossingTime,
                    p_Params.durationYears,
                    p_Params.suppliedMassFraction,
                    true
                );
            }
        }
    }

    return MakeResultFromState(state, p_Params.durationYears, p_Params.durationYears, p_Params.suppliedMassFraction, false);
}

/*
 * Map a COMPAS-labelled AIC target onto the ordered Spindler component label
 *
 * Spindler orders the components by initial mass so that q = M2/M1 <= 1,
 * whereas the public CBD interface preserves the original COMPAS star labels.
 * The mass ordering matters here only because it determines whether those labels
 * were swapped during translation; the AIC criterion itself is not mass-order dependent.
 *
 * OrderedComponent AicTargetStarToOrderedComponent(const CircumbinaryDisk::StarLabel p_AicTargetStar,
 *                                                  const bool                         p_ComponentsSwapped)
 *
 * @param   [IN]    p_AicTargetStar            COMPAS-labelled component to monitor for AIC
 * @param   [IN]    p_ComponentsSwapped        Whether COMPAS star labels were swapped to obtain Spindler ordering
 * @return                                      Ordered component monitored by the ODE, or NONE
 */
OrderedComponent AicTargetStarToOrderedComponent(
    const CircumbinaryDisk::StarLabel p_AicTargetStar,
    const bool p_ComponentsSwapped
) {
    switch (p_AicTargetStar) {
        case CircumbinaryDisk::StarLabel::NONE:
            return OrderedComponent::NONE;
        case CircumbinaryDisk::StarLabel::STAR_1:
            return p_ComponentsSwapped ? OrderedComponent::SECONDARY : OrderedComponent::PRIMARY;
        case CircumbinaryDisk::StarLabel::STAR_2:
            return p_ComponentsSwapped ? OrderedComponent::PRIMARY : OrderedComponent::SECONDARY;
    }

    return OrderedComponent::NONE;
}

/*
 * Translate COMPAS-labelled input to Spindler variables, evolve it, and translate the result back
 *
 * CircumbinaryDisk::EvolutionTrack EvolveTrackWithBackend(const CircumbinaryDisk::EvolutionInput& p_Input,
 *                                                         SpindlerBackend&                         p_Spindler)
 *
 * @param   [IN]    p_Input                    COMPAS-labelled physical input for one binary-CBD interaction segment
 * @param   [IN]    p_Spindler                 Initialised Spindler table backend
 * @return                                      Final COMPAS-labelled state and optional AIC-stop metadata
 */
CircumbinaryDisk::EvolutionTrack EvolveTrackWithBackend(
    const CircumbinaryDisk::EvolutionInput& p_Input,
    SpindlerBackend& p_Spindler
) {
    const double binaryMass = p_Input.mass1Msol + p_Input.mass2Msol;
    const bool componentsSwapped = utils::Compare(p_Input.mass1Msol, p_Input.mass2Msol) < 0;
    const double primaryMass = componentsSwapped ? p_Input.mass2Msol : p_Input.mass1Msol;
    const double secondaryMass = componentsSwapped ? p_Input.mass1Msol : p_Input.mass2Msol;

    // Convert the public COMPAS-labelled, physical-unit input into the ordered,
    // dimensionless variables expected by the Spindler ODE. componentsSwapped
    // records whether this translation exchanged the COMPAS labels; it is used
    // both to map the AIC target into ODE coordinates and to restore output labels.
    SpindlerEvolutionParams params{};
    params.suppliedMassFraction = p_Input.suppliedMassMsol / binaryMass;
    params.q0 = secondaryMass / primaryMass;
    params.e0 = p_Input.eccentricity;
    params.initialTotalMass = binaryMass;
    params.durationYears = p_Input.durationYears;
    params.stellarType1 = componentsSwapped ? p_Input.stellarType2 : p_Input.stellarType1;
    params.stellarType2 = componentsSwapped ? p_Input.stellarType1 : p_Input.stellarType2;
    params.radius1 = componentsSwapped ? p_Input.radius2Rsol : p_Input.radius1Rsol;
    params.radius2 = componentsSwapped ? p_Input.radius1Rsol : p_Input.radius2Rsol;
    params.options = p_Input.options;
    params.stopComponent = AicTargetStarToOrderedComponent(p_Input.aicTargetStar, componentsSwapped);

    const SpindlerEvolutionResult spindlerResult{p_Spindler.Evolve(params)};

    CircumbinaryDisk::EvolutionTrack track{};
    track.semiMajorAxisFactor = spindlerResult.semiMajorAxisFactor;
    track.eccentricity = spindlerResult.eccentricity;
    track.mass1Msol = (componentsSwapped ? spindlerResult.mass2 : spindlerResult.mass1) * binaryMass;
    track.mass2Msol = (componentsSwapped ? spindlerResult.mass1 : spindlerResult.mass2) * binaryMass;

    if (spindlerResult.stoppedAtMassThreshold) {
        track.aicOccurred = true;
        track.elapsedTimeYears = spindlerResult.elapsedTimeYears;
        track.suppliedMassMsol = spindlerResult.suppliedMassFractionUsed * binaryMass;
    }

    return track;
}

}  // namespace

namespace CircumbinaryDisk {

/*
 * Evolve a formed CBD and return COMPAS-labelled final state
 *
 * spindler-c is initialized lazily on the first CBD call in a COMPAS process
 * and then reused for all subsequent binaries in that process. Spindler evolves
 * a/a0, so the returned semi-major-axis value is a multiplicative factor that
 * the caller applies to its current binary separation.
 *
 * EvolutionTrack EvolveTrack(const EvolutionInput& p_Input)
 *
 * @param   [IN]    p_Input                    COMPAS-labelled physical input; per-call supplied mass is assumed to be << binary mass
 * @return                                      Final COMPAS-labelled state and optional AIC-stop metadata
 */
EvolutionTrack EvolveTrack(const EvolutionInput& p_Input) {
    static SpindlerBackend spindler{"Siwek23"};
    return EvolveTrackWithBackend(p_Input, spindler);
}


}  // namespace CircumbinaryDisk
