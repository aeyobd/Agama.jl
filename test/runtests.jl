using Agama
using Test

@testset "Potential" begin
    pot = Potential(type="NFW")

    @test pot isa Potential
    @test density(pot, [0, 0, 0]) === Inf
    pos = [1 0
           2 0
           0 1]
    @test density(pot, pos) != [1.0 / 16, 2/16] 

    @test acceleration(pot, [0, 0, 0]) ≈ zeros(3)
    pos = [1 0 
         0 -0.3
         0 0.4 ]

    @test acceleration(pot, pos) != zeros(3, 2)

    @test enclosed_mass(pot, 0) ≈ 0
    @test enclosed_mass(pot, [1, 3, 9.5]) != 0 # TODO

    @test Agama.circular_velocity(pot, 0) ≈ NaN nans=true
    @test Agama.circular_velocity(pot, [1, 3, 9.5]) != 0 # TODO


    @test potential(pot, zeros(3)) ≈ -1
    @test potential(pot, pos) != zeros(2)  # TODO


    @test Agama.stress(pot, zeros(3)) != zeros(6) # TODO
    @test Agama.stress(pot, pos) != zeros(2, 6)  # TODO
end


@testset "orbit" begin
    pot = Potential(type="Plummer")

    pos = [1,2,3]
    vel = [0,1,0]

    o = orbit(pot, pos, vel, timerange=(0, 10))
    @test o isa Agama.Orbit



    pos = [1 0
           0 3
           2 0]

    vel = [0. -0.1
           0. 0.
           -0.1 0.1]

    o = orbit(pot, pos, vel, timerange=(0, 3))
    @test o isa Vector{Agama.Orbit}


    o = orbit(pot, pos, vel, timerange=(0, -1.2))
    @test o isa Vector{Agama.Orbit}
end


@testset "actions" begin
    pot = Potential(type="NFW")

    pos = [1, 0, 0]
    vel = [0, 0.1, 0]
    am = ActionMapper(pot)
    af = ActionFinder(pot)

    act = actions(af, pos, vel)
    act2, ang, freq = actions_angles(af, pos, vel)
    pos2, vel2 = from_actions(am, act, ang)

    @test pos2 ≈ pos atol=1e-8
    @test vel2 ≈ vel atol=1e-8
    @test act2 ≈ act 

    pos = rand(3, 4)
    vel = rand(3, 4) / 5
    am = ActionMapper(pot)
    af = ActionFinder(pot)

    act = actions(af, pos, vel)
    act2, ang, freq = actions_angles(af, pos, vel)
    pos2, vel2 = from_actions(am, act, ang)

    @test pos2 ≈ pos atol=1e-6
    @test vel2 ≈ vel atol=1e-6
    @test act2 ≈ act
end



@testset "galaxymodel" begin
    pot = Potential(type="NFW")
    df = DistributionFunction(pot, type="QuasiSpherical")
    gm = GalaxyModel(pot, df)
    pos, vel = sample(gm, 100)
    @test size(pos) == (3, 100)
    @test size(vel) == (3, 100)
end
