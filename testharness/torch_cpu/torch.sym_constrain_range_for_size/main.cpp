#include "fuzzer_utils.h"
#include <ATen/ops/sym_constrain_range_for_size.h>
#include <iostream>
#include <cstring>
#include <optional>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        // Need at least some bytes for parameters
        if (Size < 8) {
            return 0;
        }

        size_t offset = 0;

        // Extract the size value to constrain (must be positive for "size" semantics)
        int64_t size_val = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&size_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        // Make size positive (sizes are typically positive)
        size_val = std::abs(size_val);
        if (size_val == 0) size_val = 1; // Avoid zero

        at::Scalar size_scalar(size_val);

        // Extract min value (defaults to 0 for "for_size" variant)
        int64_t min_val = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // min should be >= 0 for size constraints
            min_val = std::abs(min_val);
        }

        // Extract max value (must be > 2 if provided)
        int64_t max_val = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            // max must be > 2 according to implementation
            max_val = std::abs(max_val);
            if (max_val <= 2) {
                max_val = 3; // Make it valid
            }
        }

        // Ensure valid range: min <= max
        if (min_val > max_val) {
            std::swap(min_val, max_val);
            // After swap, ensure max is still > 2
            if (max_val <= 2) {
                max_val = 3;
            }
        }

        // Case 1: Call with no optional arguments (uses defaults: min=0, max=nullopt)
        try {
            at::sym_constrain_range_for_size(at::Scalar(static_cast<int64_t>(1)));
        } catch (...) {
            // Expected for invalid ranges, silently ignore
        }

        // Case 2: Call with only min specified
        try {
            // Make sure size >= min for this to work
            int64_t valid_size = std::max(size_val, min_val);
            at::sym_constrain_range_for_size(at::Scalar(valid_size), min_val);
        } catch (...) {
            // Silently ignore expected failures
        }

        // Case 3: Call with both min and max specified, ensuring size is in range
        try {
            int64_t constrained_size = size_val;
            // Clamp size to be within [min_val, max_val]
            if (constrained_size < min_val) constrained_size = min_val;
            if (constrained_size > max_val) constrained_size = max_val;
            at::sym_constrain_range_for_size(at::Scalar(constrained_size), min_val, max_val);
        } catch (...) {
            // Silently ignore expected failures
        }

        // Case 4: Try with size at boundary conditions
        try {
            // Size at min boundary
            at::sym_constrain_range_for_size(at::Scalar(min_val), min_val, max_val);
        } catch (...) {
            // Silently ignore
        }

        try {
            // Size at max boundary
            at::sym_constrain_range_for_size(at::Scalar(max_val), min_val, max_val);
        } catch (...) {
            // Silently ignore
        }

        // Case 5: Call with nullopt for max (unbounded upper limit)
        try {
            int64_t valid_size = std::max(size_val, min_val);
            at::sym_constrain_range_for_size(at::Scalar(valid_size), min_val, std::nullopt);
        } catch (...) {
            // Silently ignore
        }

        // Case 6: Use fuzzer-controlled values directly (may throw, that's expected)
        try {
            at::sym_constrain_range_for_size(size_scalar, min_val, max_val);
        } catch (...) {
            // Silently ignore - this tests invalid input handling
        }

        // Case 7: Test with very large values
        try {
            int64_t large_max = std::numeric_limits<int64_t>::max();
            at::sym_constrain_range_for_size(at::Scalar(static_cast<int64_t>(100)), 0, large_max);
        } catch (...) {
            // Silently ignore
        }

        // Case 8: Test with optional parameters explicitly set to nullopt
        try {
            at::sym_constrain_range_for_size(at::Scalar(static_cast<int64_t>(5)), std::nullopt, std::nullopt);
        } catch (...) {
            // Silently ignore
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}