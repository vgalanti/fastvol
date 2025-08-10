/**
 * @file fastvol.hpp
 * @brief Main header for the fastvol library.
 *
 * @author Valerio Galanti
 * @date 2025
 * @version 0.1.1
 * @license MIT License
 *
 * This file includes all the necessary headers for the fastvol library and provides
 * top-level utility functions.
 */

#pragma once

#include "american/bopm.hpp"
#include "american/psor.hpp"
#include "american/ttree.hpp"
#include "european/bsm.hpp"

/**
 * @brief Returns the version of the fastvol library.
 * @return A string representing the library version.
 */
const char *fastvol_version(void) { return "0.1.1"; }

/**
 * @brief Checks if the fastvol library was compiled with CUDA support.
 * @return True if CUDA is available, false otherwise.
 */
bool fastvol_cuda_available(void)
{
#ifdef FASTVOL_CUDA_ENABLED
    return true;
#else
    return false;
#endif
}
