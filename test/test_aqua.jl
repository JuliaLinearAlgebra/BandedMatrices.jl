module AquaTest

import BandedMatrices
import Aqua
using Test

@testset "Project quality" begin
    Aqua.test_all(BandedMatrices, ambiguities=false, piracies=false)
end

end
