module AquaTest

import BandedMatrices
import Aqua
using Test
using ArrayLayouts: AbstractBandedLayout, checkdimensions, DiagonalLayout,
                    BidiagonalLayout, TriangularLayout, AbstractTridiagonalLayout

@testset "Project quality" begin
    Aqua.test_all(BandedMatrices, ambiguities=false,
        piracies=(; treat_as_own=Union{Function, Type}[AbstractBandedLayout, checkdimensions, DiagonalLayout,
                    BidiagonalLayout, TriangularLayout, AbstractTridiagonalLayout], broken=true)
    )
end

end
