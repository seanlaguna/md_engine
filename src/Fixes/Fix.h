#pragma once
#ifndef FIX_H
#define FIX_H

#include "Bond.h"
#include "globalDefs.h"
#include <pugixml.hpp>

class Atom;
class State;

//! Make class Fix available to Python interface
void export_Fix();

//! Base class for Fixes
/*!
 * Fixes modify the dynamics in the system. They are called by the Integrator
 * at regular intervals and they can modify the Atom forces, positions and
 * velocities. Note that as Fixes themselves depend on the current forces,
 * positions and velocities, the order in which the Fixes are defined and
 * called is important.
 *
 * For this reason, Fixes have an order preference. Fixes with a low preference
 * are computed first.
 *
 * \todo Compile list of preferences. I think each Fix should have an order
 *       preference.
 */
class Fix {

protected:
    //! Default constructor
    /*!
     * Delete default constructor.
     */
    Fix() = delete;

    //! Constructor
    /*!
     * \param state_ Pointer to simulation state
     * \param handle_ Name of the Fix
     * \param groupHandle_ String specifying on which group the Fix acts
     * \param type_ Type of the Fix (unused?)
     * \param applyEvery_ Apply Fix every this many timesteps
     *
     * \todo Make constructor protected since this is an abstract base class
     */
    Fix(boost::shared_ptr<State> state_, std::string handle_, std::string groupHandle_,
        std::string type_, bool forceSingle_, int applyEvery_,
        int orderPreference_ = 0);

public:
    //! Destructor
    virtual ~Fix() {};

    //! Test if another Fix is the same
    /*!
     * \param f Reference of Fix to test
     * \return True if they have the same handle
     *
     * Two Fixes are considered equal if they have the same "name" stored in
     * handle. Thus, a FixLJCut called "myFix" and a FixSpringStatic called
     * "myFix" are considered equal.
     *
     * \todo Take const reference, make function const
     * \todo Why not use operator==()?
     * \todo Isn't comparing the handle overly simplistic?
     */
    bool isEqual(Fix &f);

    //! Prepare Fix for run
    /*!
     * \return False if a problem occured, else True
     */
    virtual bool prepareForRun() {return true;};

    //! Perform calculations at the end of a simulation run
    /*!
     * \return False if an error occured, else return true
     *
     * Some Fixes set up internal variables in the Fix::prepareForRun()
     * function. This function then typically sets these values back to their
     * default.
     *
     * \todo Make this function purely virtual
     */
    virtual bool postRun() { return true; }

    //! Apply fix
    /*!
     * \param computeVirials Compute virials for this Fix
     *
     * \todo Make purely virtual.
     */
    virtual void compute(bool computeVirials) {}

    //! Calculate single point energy of this Fix
    /*!
     * \param perParticleEng Pointer to where to store the per-particle energy
     *
     * The pointer passed needs to be a pointer to a memory location on the
     * GPU.
     *
     * \todo Use cuPointerGetAttribute() to check that the pointer passed is a
     *       pointer to GPU memory.
     *
     * \todo Make this function purely virtual.
     */
    virtual void singlePointEng(float *perParticleEng) {}

    //! Accomodate for new type of Atoms added to the system
    /*!
     * \param handle String specifying the new type of Atoms
     *
     * \todo Make purely virtual.
     */
    virtual void addSpecies(std::string handle) {}

    //! Recalculate group bitmask from a (possibly changed) handle
    void updateGroupTag();

    //! Refresh Atoms
    /*!
     * \return False if a problem occured, else True
     *
     * This function should be called whenever the number of atoms in the
     * simulation has changed.
     */
    virtual bool refreshAtoms(){return true;};

    //! Restart Fix
    /*!
     * \param restData XML node containing the restart data for the Fix
     *
     * \return False if restart data could not be loaded, else return True
     *
     * Setup Fix from restart data.
     */
    virtual bool readFromRestart(pugi::xml_node restData){return true;};

    //! Write restart data
    /*!
     * \param format Format for restart data
     *
     * \return Restart string
     *
     * Write out information of this Fix to be reloaded via
     * Fix::readFromRestart().
     */
    virtual std::string restartChunk(std::string format){return "";};

    //! Return list of Bonds
    /*!
     * \return Pointer to list of Bonds or nullptr if Fix does not handle Bonds
     *
     * \todo Think about treatment of different kinds of bonds in fixes right
     *       now for ease, each vector of bonds in any given fix that stores
     *       bonds has to store them in a vector<BondVariant> variable you can
     *       push_back, insert, whatever, other kinds of bonds into this vector
     *       you have to get them out using a getBond method, or using the
     *       boost::get<BondType>(vec) syntax. It's not perfect, but it lets us
     *       generically collect vectors without doing any copying.
     */
    virtual std::vector<BondVariant> *getBonds() {
        return nullptr;
    }

    //! Return list of cutoff values.
    /*!
     * \return vector storing interaction cutoff values or empty list if no
     *         cutoffs are used.
     */
    virtual std::vector<float> getRCuts() {
        return std::vector<float>();
    }

    //! Check that all given Atoms are valid
    /*!
     * \param atoms List of Atom pointers
     *
     * This function verifies that all Atoms to be tested are valid using the
     * State::validAtom() method. The code crashes if an invalid Atom is
     * encountered.
     *
     * \todo A crash is not a very graceful method of saying that an Atom was
     *       invalid.
     * \todo Pass const reference. Make this function const.
     */
    void validAtoms(std::vector<Atom *> &atoms);

public:
    State *state; //!< Pointer to the simulation state
    std::string handle; //!< "Name" of the Fix
    std::string groupHandle; //!< Group to which the Fix applies
    const std::string type; //!< String naming the Fix type
    int applyEvery; //!< Applyt this fix every this many timesteps
    unsigned int groupTag; //!< Bitmask for the group handle

    const bool forceSingle; //!< True if Fix contributes to single point energy.
    int orderPreference; //!< Fixes with a high order preference are calculated
                         //!< later.

    const std::string restartHandle; //!< Handle for restart string
};

/*
do it with precompiler instructions, lol!
nah, just do methods of state.  Might have to add other function calls later as fixes become more complicated
*/
#endif
