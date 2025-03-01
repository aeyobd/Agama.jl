module Agama

using PythonCall

export py2f, py2vec, py2mat
export Potential, DistributionFunction, GalaxyModel
export agama
export sample
export calc_Φ, calc_acceleration, calc_ρ
export orbit, Orbit


const agama = Ref{Py}()
const np = Ref{Py}()

F = Float64


function __init__()
    agama[] = pyimport("agama")
    np[] = pyimport("numpy")
end


"""
    py2f(x::Py)::F

Converts a python object to a Float64
"""
function py2f(x::Py)::F
    return pyconvert(F, x)
end



py2vec(x::Py)::Vector{F} = pyconvert(Vector{F}, x)
py2mat(x::Py)::Matrix{F} = pyconvert(Matrix{F}, x)'

mat2py(x::AbstractMatrix{<:Real})::Py = np[].array(x')


struct Potential
    _py::Py
end

function Potential(; kwargs...)
    return Potential(agama[].Potential(;kwargs...))
end


"""
    calc_Φ(potential::Potential, pos::AbstractMatrix{<:Real}; kwargs...)

Given a 3xN matrix of positions, returns the potential at each position. 
Optionally can specify `t`. 
"""
function calc_Φ(potential::Potential, pos::AbstractMatrix{<:Real}; kwargs...)
    return potential._py.potential(mat2py(pos); kwargs...) |> py2vec
end


"""
    calc_Φ(potential::Potential, pos::AbstractVector{<:Real}; kwargs...)

Given a 3 vector of positions, returns the potential at that position.
Optionally can specify `t`.
"""
function calc_Φ(potential::Potential, pos::AbstractVector{<:Real}; kwargs...)
    return potential._py.potential(pos; kwargs...) |> py2f
end


"""
    calc_acceleration(potential::Potential, pos::AbstractMatrix{<:Real}; kwargs...)

Given a 3xN matrix of positions, returns the acceleration at each position.
Optionally can specify `t`.
"""
function calc_acceleration(potential::Potential, pos::AbstractMatrix{<:Real}; kwargs...)
    return potential._py.force(mat2py(pos); kwargs...) |> py2mat
end


"""
    calc_acceleration(potential::Potential, pos::AbstractVector{<:Real}; kwargs...)

Given a 3 vector of positions, returns the acceleration at that position.
Optionally can specify `t`.
"""
function calc_acceleration(potential::Potential, pos::AbstractVector{<:Real}; kwargs...)
    return potential._py.force(pos; kwargs...) |> py2vec
end


"""
    calc_ρ(potential::Potential, pos::AbstractMatrix{<:Real}; kwargs...)

Given a 3xN matrix of positions, returns the density at each position.
Optionally can specify `t`.
"""
function calc_ρ(potential::Potential, pos::AbstractMatrix{<:Real}; kwargs...)
    return potential._py.density(mat2py(pos); kwargs...) |> py2vec
end


"""
    calc_ρ(potential::Potential, pos::AbstractVector{<:Real}; kwargs...)

Given a 3 vector of positions, returns the density at that position.
Optionally can specify `t`.
"""
function calc_ρ(potential::Potential, pos::AbstractVector{<:Real}; kwargs...)
    return potential._py.density(pos; kwargs...) |> py2f
end


struct DistributionFunction
    _py::Py
end


function DistributionFunction(; kwargs...)
    return DistributionFunction(agama[].DistributionFunction(;kwargs...))
end



struct GalaxyModel
    _py::Py
end


function GalaxyModel(potential::Potential, df::DistributionFunction; kwargs...)
    return GalaxyModel(agama[].GalaxyModel(potential._py, df._py; kwargs...))
end


"""
    sample(gm::GalaxyModel, N::Int)

Samples the galaxy model and returns the positions, velocities and masses of the particles
"""
function sample(gm::GalaxyModel, N::Int)
    posvel, mass = gm._py.sample(N)
    posvel = py2mat(posvel)
    mass = py2vec(mass)

    return (posvel[1:3, :], posvel[4:6, :], mass)
end


struct Orbit

end


function orbit(; kwargs...)
    pyorbit = agama[].orbit(; kwargs...)
    return pyorbit
end



# Documentation

"""
    Potential(; kwargs...)

Create an Agama potential object. 
Thin wrapper around python interface

# Agama documentation
Potential is a class that represents a wide range of gravitational potentials.
  There are several ways of initializing the potential instance:
    - from a list of key=value arguments that specify an elementary potential class;
    - from a tuple of dictionary objects that contain the same list of possible key/value pairs for each component of a composite potential;
    - from an INI file with these parameters for one or several components and/or with potential expansion coefficients previously stored by the export() method;
    - from a tuple of existing Potential objects created previously (in this case a composite potential is created from these components).
  Note that all keywords and their values are not case-sensitive.
  
  List of possible keywords for a single component:
    type='...'   the type of potential, can be one of the following 'basic' types:
      Harmonic, Logarithmic, Plummer, MiyamotoNagai, NFW, Ferrers, Dehnen, PerfectEllipsoid, Disk, Spheroid, Nuker, Sersic, King, KeplerBinary, UniformAcceleration;
      or one of the expansion types:  BasisSet, Multipole, CylSpline - in these cases, one should provide either a density model, file name, or an array of particles.
    mass=...   total mass of the model, if applicable.
    scaleRadius=...   scale radius of the model (if applicable).
    scaleHeight=...   scale height of the model (currently applicable to MiyamotoNagai and Disk).
    p=...   or  axisRatioY=...   axis ratio y/x, i.e., intermediate to long axis (applicable to triaxial potential models such as Dehnen and Ferrers, and to Spheroid, Nuker or Sersic density models).
    q=...   or  axisRatioZ=...   short to long axis (z/x) (applicable to the same model types as above plus the axisymmetric PerfectEllipsoid).
    gamma=...  central cusp slope (applicable for Dehnen, Spheroid or Nuker).
    beta=...   outer density slope (Spheroid or Nuker).
    alpha=...  strength of transition from the inner to the outer slopes (Spheroid or Nuker).
    sersicIndex=...   profile shape parameter 'n' (Sersic or Disk).
    innerCutoffRadius=...   radius of inner hole (Disk).
    outerCutoffRadius=...   radius of outer exponential cutoff (Spheroid).
    cutoffStrength=...   strength of outer exponential cutoff  (Spheroid).
    surfaceDensity=...   surface density normalization (Disk or Sersic - in the center, Nuker - at scaleRadius).
    densityNorm=...   normalization of density profile (Spheroid).
    W0=...  dimensionless central potential in King models.
    trunc=...  truncation strength in King models.
    center=...  offset of the potential from origin, can be either a triplet of numbers, or an array of time-dependent offsets (t,x,y,z, and optionally vx,vy,vz) provided directly or as a file name.
    orientation=...  orientation of the principal axes of the model w.r.t. the external coordinate system, specified as a triplet of Euler angles.
    rotation=...  angle of rotation of the model about its z axis, can be a single number or an array / file with a time-dependent angle.
    scale=...  modification of mass and size scales of the model, given either as two numbers or an array / file with time-dependent scaling factors.
  Parameters for potential expansions:
    density=...   the density model for a potential expansion.
    It may be a string with the name of density profile (most of the elementary potentials listed above can be used as density models, except those with infinite mass; in addition, there are other density models without a corresponding potential).
    Alternatively, it may be an object providing an appropriate interface -- either an instance of Density or Potential class, or a user-defined function `my_density(xyz)` returning the value of density computed simultaneously at N points, where xyz is a Nx3 array of points in cartesian coordinates (even if N=1, it's a 2d array).
    potential=...   instead of density, one may provide a potential source for the expansion. This argument shoud be either an instance of Potential class, or a user-defined function `my_potential(xyz)` returning the value of potential at N point, where xyz is a Nx3 array of points in cartesian coordinates. 
    file='...'   the name of another INI file with potential parameters and/or coefficients of a Multipole/CylSpline potential expansion, or an N-body snapshot file that will be used to compute the coefficients of such expansion.
    particles=(coords, mass)   array of point masses to be used in construction of a potential expansion (an alternative to density=..., potential=... or file='...' options): should be a tuple with two arrays - coordinates and mass, where the first one is a two-dimensional Nx3 array and the second one is a one-dimensional array of length N.
    symmetry='...'   assumed symmetry for potential expansion constructed from an N-body snapshot or from a user-defined density or potential function (required in these cases). Possible options, in order of decreasing symmetry: 'Spherical', 'Axisymmetric', 'Triaxial', 'Bisymmetric', 'Reflection', 'None', or a numerical code; only the case-insensitive first letter matters).
    gridSizeR=...   number of radial grid points in Multipole and CylSpline potentials.
    gridSizeZ=...   number of grid points in z-direction for CylSpline potential.
    rmin=...   radius of the innermost grid node for Multipole and CylSpline; zero(default) means auto-detect.
    rmax=...   same for the outermost grid node.
    zmin=...   z-coordinate of the innermost grid node in CylSpline (zero means autodetect).
    zmax=...   same for the outermost grid node.
    lmax=...   order of spherical-harmonic expansion (max.index of angular harmonic coefficient) in Multipole.
    mmax=...   order of azimuthal-harmonic expansion (max.index of Fourier coefficient in phi angle) in Multipole and CylSpline.
    smoothing=...   amount of smoothing in Multipole initialized from an N-body snapshot.
    nmax=...   order of radial expansion in BasisSet (the number of basis functions is nmax+1).
    eta=...    shape parameter of basis functions in BasisSet (default is 1.0, corresponding to the Hernquist-Ostriker basis set, but values up to 2.0 typically provide better accuracy for cuspy density profiles; the minimum value is 0.5, corresponding to the Clutton-Brock basis set.
    r0=...     scale radius of basis functions in BasisSet; if not provided, will be assigned automatically to the half-mass radius, unless the model has infinite mass.
  
  Most of these parameters have reasonable default values; the only necessary ones are `type`, and for a potential expansion, `density` or `file` or `particles`.
  If the parameters of the potential (including the coefficients of a potential expansion) areloaded from a file, then the `type` argument should not be provided, and the argument name `file=` may be omitted (i.e., may provide only the filename as an unnamed string argument).
  One may create a modified version of an existing Potential object, by passing it as a `potential` argument together with one or more modifier parameters (center, orientation, rotation and scale); in this case, `type` should be empty.
  Examples:
  
  >>> pot_halo = Potential(type='Dehnen', mass=1e12, gamma=1, scaleRadius=100, p=0.8, q=0.6)
  >>> pot_disk = Potential(type='MiyamotoNagai', mass=5e10, scaleRadius=5, scaleHeight=0.5)
  >>> pot_composite = Potential(pot_halo, pot_disk)
  >>> pot_from_ini = Potential('my_potential.ini')
  >>> pot_from_snapshot = Potential(type='Multipole', file='snapshot.dat')
  >>> pot_from_particles = Potential(type='Multipole', particles=(coords, masses), symmetry='t')
  >>> pot_user = Potential(lambda x: -(numpy.sum(x**2, axis=1) + 1)**-0.5, symmetry='s')
  >>> pot_shifted = Potential(potential=pot_composite, center=[1.0,2.0,3.0]
  >>> dens_func = lambda xyz: 1e8 / (numpy.sum((xyz/10.)**4, axis=1) + 1)
  >>> pot_exp = Potential(type='Multipole', density=dens_func, symmetry='t', gridSizeR=20, Rmin=1, Rmax=500, lmax=4)
  >>> disk_par = dict(type='Disk', surfaceDensity=1e9, scaleRadius=3, scaleHeight=0.4)
  >>> halo_par = dict(type='Spheroid', densityNorm=2e7, scaleRadius=15, gamma=1, beta=3, outerCutoffRadius=150, axisRatioZ=0.8)
  >>> pot_galpot = Potential(disk_par, halo_par)
  
  The latter example illustrates the use of GalPot components (exponential disks and spheroids) from Dehnen&Binney 1998; these are internally implemented using a Multipole potential expansion and a special variant of disk potential, but may also be combined with any other components if needed.
  The numerical values in the above examples are given in solar masses and kiloparsecs; a call to `setUnits(length=1, mass=1, velocity=1)` should precede the construction of potentials in this approach. Alternatively, one may provide no units at all, and use the `N-body` convention G=1 (this is the default regime and is restored by calling `setUnits()` without arguments).
"""
function Potential
end


"""
    DistributionFunction(; kwargs...)

Create an Agama distribution function object.

# Agama documentation

  DistributionFunction class represents an action-based distribution function.
  
  The constructor accepts several key=value arguments that describe the parameters of distribution function.
  Required parameter is type='...', specifying the type of DF. Currently available types are:
  'DoublePowerLaw' (for the halo);
  'QuasiIsothermal' and 'Exponential' (for the disk component);
  'QuasiSpherical' (for the isotropic or anisotropic DF of the Cuddeford-Osipkov-Merritt type corresponding to a given density profile - by default it is the isotropic DF produced by the Eddington inversion formula).
  For some of them, one also needs to provide the potential to initialize the table of epicyclic frequencies (potential=... argument). For the QuasiSpherical DF one needs to provide an instance of density profile (density=...) and the potential (if they are the same, then only potential=... is needed), and optionally the central value of anisotropy coefficient `beta0` (by default 0) and the anisotropy radius `r_a` (by default infinity).
  Other parameters are specific to each DF type.
  Alternatively, a composite DF may be created from an array of previously constructed DFs:
  >>> df = DistributionFunction(df1, df2, df3)
  
  The () operator computes the value of distribution function for the given triplet of actions, or N such values if the input is a 2d array of shape Nx3. When called with an optional argument der=True, it returns a 2-tuple with the DF values (array of length N) and its derivatives w.r.t. actions (array of shape Nx3).
  The totalMass() function computes the total mass in the entire phase space.
  
  One may provide a user-defined DF function in all contexts where a DistributionFunction object is required. This function should take a single positional argument - Nx3 array of actions (with columns representing Jr, Jz, Jphi at N>=1 points) and returns an array of length N. This function may optionally provide derivatives when called with a named argument der=True, and in this case should return a 2-tuple with DF values (array of length N) and derivatives (array of shape Nx3).

"""
function DistributionFunction end



"""
    orbit(; kwargs...)

Create an orbit object. Wrapper around python interface.

# Agama documentation
Compute a single orbit or a bunch of orbits in the given potential.
  Named arguments:
    ic:  initial conditions - either an array of 6 numbers (3 positions and 3 velocities in Cartesian coordinates) for a single orbit, or a 2d array of Nx6 numbers for a bunch of orbits.
    potential:  a Potential object or a compatible interface.
    Omega (optional, default 0):  pattern speed of the rotating frame.
    time:  total integration time - may be a single number (if computing a single orbit or if it is identical for all orbits), or an array of length N (for a bunch of orbits).
    timestart (optional, default 0):  initial time for the integration (only matters if the potential is time-dependent). The final time is thus timestart+time. May be a single number (for one orbit or if it is identical for all orbits), or an array of length N (for a bunch of orbits).
    targets (optional):  zero or more instances of Target class (a tuple/list if more than one); each target collects its own data for each orbit.
    trajsize (optional):  if given, turns on the recording of trajectory for each orbit (should be either a single integer or an array of integers with length N). The trajectory of each orbit is stored either at every timestep of the integrator (if trajsize=0) or at regular intervals of time (`dt=abs(time)/(trajsize-1)`, so that the number of points is `trajsize`; the last stored point is always at the end of integration period, and if trajsize>1, the first point is the initial conditions). If dtype=object and trajsize is not provided explicitly, this is equivalent to setting trajsize=0. Both time and trajsize may differ between orbits.
    der (optional, default False):  whether to compute the evolution of deviation vectors (derivatives of the orbit w.r.t. the initial conditions).
    lyapunov (optional, default False):  whether to estimate the Lyapunov exponent, which is a chaos indicator (positive value means that the orbit is chaotic, zero - regular).
    accuracy (optional, default 1e-8):  relative accuracy of the ODE integrator.
    maxNumSteps (optional, default 1e8):  upper limit on the number of steps in the ODE integrator.
    dtype (optional, default 'float32'):  storage data type for trajectories (see below).
    method (optional, string):  choice of the ODE integrator, available variants are 'dop853' (default; 8th order Runge-Kutta) or 'hermite' (4th order, may be more efficient in the regime of low accuracy).
    verbose (optional, default True):  whether to display progress when integrating multiple orbits.
  Returns:
    depending on the arguments, one or a tuple of several data containers (one for each target, plus an extra one for trajectories if trajsize>0, plus another one for deviation vectors if der=True, plus another one for Lyapunov exponents if lyapunov=True). 
    Each target produces a 2d array of floats with shape NxC, where N is the number of orbits, and C is the number of constraints in the target (varies between targets); if there was a single orbit, then this would be a 1d array of length C. These data storage arrays should be provided to the `solveOpt()` routine. 
    Lyapunov exponent is a single number for one orbit, or a 1d array for several orbits.
    Trajectory output and deviation vectors can be requested in two alternative formats: arrays or Orbit objects.
  In the first case, the output of the trajectory is a Nx2 array (or, in case of a single orbit, a 1d array of length 2), with elements being objects themselves: each row stands for one orbit, the first element in each row is a 1d array of length `trajsize` containing the timestamps, and the second is a 2d array of phase-space coordinates at the corresponding timestamps, in the format depending on dtype:
  'float' or 'double' means 6 64-bit floats (3 positions and 3 velocities) in each row;
  'float32' (default) means 6 32-bit floats;
  'complex' or 'complex128' or 'c16' means 3 128-bit complex values (pairs of 64-bit floats), with velocity in the imaginary part; and 'complex64' or 'c8' means 3 64-bit complex values.
  The time array is always 64-bit float. The choice of dtype only affects trajectories; arrays returned by each target always contain 32-bit floats.
  In the second case (dtype=object), the output is a 1d array of length N containing instances of a special class agama.Orbit, or just an Orbit object for a single orbit. The agama.Orbit class can only be returned by the orbit() routine and cannot be instantiated directly. It provides interpolated trajectory at any time within the range spanned by the orbit: its () operator takes one argument (a single value of time or an array of times), and returns one or more 6d phase-space coordinates at the requested times. It also exposes a sequence interface: len(orbit) returns the number of timestamps in the interpolator, and orbit[i] returns the i-th timestamp, so that the full 6d trajectory at these timestamps can be reconstructed by applying the () operator to the orbit object itself. Although such interpolator may be constructed from a regularly-spaced orbit, it makes more sense to leave trajsize=0 in this case, i.e., record the trajectory at every timestep of the orbit integrator. This is the default behaviour, and trajsize needs not be specified explicitly when setting dtype=object.
  The output for deviation vectors, if they are requested, follows the same format as for the trajectory, except that there are 6 such vectors for each orbit. Thus, if dtype=object, each orbit produces an array of 6 agama.Orbit objects, each one representing a single deviation vector; for other dtypes, the output for one orbit consists of 6 arrays of shape `trajsize`*6 (for dtype='float' or 'double'), or `trajsize`*3 (for dtype='complex' or 'complex128'), each one representing a single deviation vector sampled at the same timestamps as the trajectory. in case of N>1 orbits, the output is an array of shape Nx6 filled with agama.Orbit objects or 2d arrays of deviation vectors.
  
  Examples:
  # compute a single orbit and output the trajectory in a 2d array of size 1001x6:
  >>> times,traj = agama.orbit(potential=mypot, ic=[x,y,z,vx,vy,vz], time=100, trajsize=1001)
  # record the same orbit at its 'natural' timestep and represent it as an agama.Orbit object:
  >>> orbit = agama.orbit(potential=mypot, ic=[x,y,z,vx,vy,vz], time=100, dtype=object)
  >>> traj_recorded = orbit(orbit)       # produces a 2d array of size len(orbit) x 6
  >>> traj_interpolated = orbit(times)   # produces an array of size 1001x6 very close to traj
  # integrate a bunch of orbits with initial conditions taken from a Nx6 array `initcond`, for a time equivalent to 50 periods for each orbit, collecting the data for two targets `target1` and `target2` and also storing their trajectories in a Nx2 array of time and position/velocity arrays:
  >>> stor1, stor2, trajectories = agama.orbit(potential=mypot, ic=initcond, time=50*mypot.Tcirc(initcond), trajsize=500, targets=(target1, target2))
  # compute a single orbit and its deviation vectors v0..v5, storing only the final values at t=tend, and estimate the Lyapunov exponent (if it is positive, the magnitude of deviation vectors grows exponentially with time, otherwise grows linearly):
  >>> (time,endpoint), (v0,v1,v2,v3,v4,v5), lyap = agama.orbit(potential=mypot, ic=[x,y,z,vx,vy,vz], time=100, trajsize=1, der=True, lyapunov=True)
"""
function orbit end


"""
    GalaxyModel(potential::Potential, df::DF; kwargs...)

Create a galaxy model object. Wrapper around python interface. 

# Agama documentation
GalaxyModel is a class that takes together a Potential, a DistributionFunction, and an ActionFinder objects, and provides methods to compute moments and projections of the distribution function at a given point in the ordinary phase space (coordinate/velocity), as well as methods for drawing samples from the distribution function in the given potential.
  The constructor takes the following arguments:
    potential - a Potential object.
    df - a DistributionFunction object.
    af (optional) - an ActionFinder object - must be constructed for the same potential; if not provided, then the action finder is created internally.
    sf (optional) - a SelectionFunction object or a user-defined callable function that takes a 2d Nx6 array of phase-space points (x,v in cartesian coordinates) as input, and returns a 1d array of N values between 0 and 1, which will be multiplied by the values of the DF at corresponding points; if not provided, assumed identically unity.
  In case of a multicomponent DF, one may compute the moments and projections for each component separately by providing an optional flag 'separate=True' to the corresponding methods. This is more efficient than constructing a separate GalaxyModel instance for each DF component and computing its moments, because the most expensive operation - conversion between position/velocity and action space - is performed once for all components. If this flag is not set (default), all components are summed up.

"""
function GalaxyModel end

end
