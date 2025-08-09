#include "fastvol/detail/bench.hpp"
#include <cstddef>

using namespace fastvol::detail::bench;

int main()
{
    fastvol::detail::bench::european::bsm::all();
    fastvol::detail::bench::american::bopm::all();
    fastvol::detail::bench::american::ttree::all();
    fastvol::detail::bench::american::psor::all();
    return 0;
}
