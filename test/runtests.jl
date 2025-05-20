using Agama
using Test

@testset "Potential" begin
    pot = Potential(type="NFW")

    @test pot isa Potential
    @test density(pot, [0, 0, 0]) === Inf
    @test density(pot, [1 0
                        2 0
                        0 1]) ≈ [1.0 / 16, 2/16] broken=true

    @test acceleration(pot, [0, 0, 0]) ≈ zeros(3)
    @test acceleration(pot, [1 0 
                             0 -0.3
                             0 0.4 ]) ≈ [-1 0 
                                  0 0
                                  0 1
                                 ] broken=true

    @test enclosed_mass(pot, 0) ≈ 0
    @test enclosed_mass(pot, [1, 3, 9.5]) ≈ 1 broken=true

    @test potential(pot, 0) ≈ -1 broken=true
    @test potential(pot, [1,2,3]) ≈ zeros(3) broken=true
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
