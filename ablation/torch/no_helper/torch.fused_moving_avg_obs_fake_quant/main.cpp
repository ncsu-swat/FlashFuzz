#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 32) return 0;

        // Extract tensor dimensions
        auto dims = extract_tensor_dims(Data, Size, offset, 4);
        if (dims.empty()) return 0;

        // Create input tensor
        auto input = create_tensor(Data, Size, offset, dims, torch::kFloat);
        if (!input.defined()) return 0;

        // Create scale tensor (single value)
        auto scale = create_tensor(Data, Size, offset, {1}, torch::kFloat);
        if (!scale.defined()) return 0;

        // Create zero_point tensor (single value)  
        auto zero_point = create_tensor(Data, Size, offset, {1}, torch::kFloat);
        if (!zero_point.defined()) return 0;

        // Create running_min tensor (single value)
        auto running_min = create_tensor(Data, Size, offset, {1}, torch::kFloat);
        if (!running_min.defined()) return 0;

        // Create running_max tensor (single value)
        auto running_max = create_tensor(Data, Size, offset, {1}, torch::kFloat);
        if (!running_max.defined()) return 0;

        // Extract averaging_constant
        double averaging_constant = extract_float_in_range(Data, Size, offset, 0.001, 0.999);

        // Extract quant_min and quant_max
        int64_t quant_min = extract_int_in_range(Data, Size, offset, -128, 0);
        int64_t quant_max = extract_int_in_range(Data, Size, offset, quant_min + 1, 255);

        // Extract ch_axis (channel axis)
        int64_t ch_axis = extract_int_in_range(Data, Size, offset, -input.dim(), input.dim() - 1);

        // Extract per_row_fake_quant flag
        bool per_row_fake_quant = extract_bool(Data, Size, offset);

        // Extract symmetric_quant flag  
        bool symmetric_quant = extract_bool(Data, Size, offset);

        // Ensure running_min <= running_max
        auto min_val = torch::min(running_min, running_max);
        auto max_val = torch::max(running_min, running_max);
        running_min = min_val;
        running_max = max_val;

        // Test basic functionality
        auto result = torch::fused_moving_avg_obs_fake_quant(
            input, 
            scale, 
            zero_point, 
            running_min, 
            running_max, 
            averaging_constant,
            quant_min,
            quant_max,
            ch_axis,
            per_row_fake_quant,
            symmetric_quant
        );

        // Verify result is a tuple with expected number of elements
        if (std::get<0>(result).defined()) {
            auto fake_quantized = std::get<0>(result);
            // Basic sanity checks
            if (fake_quantized.sizes() != input.sizes()) {
                throw std::runtime_error("Output size mismatch");
            }
        }

        // Test edge cases with different tensor shapes
        if (input.numel() > 0) {
            // Test with different channel axes
            for (int64_t axis = -input.dim(); axis < input.dim(); axis++) {
                try {
                    auto edge_result = torch::fused_moving_avg_obs_fake_quant(
                        input, 
                        scale, 
                        zero_point, 
                        running_min, 
                        running_max, 
                        averaging_constant,
                        quant_min,
                        quant_max,
                        axis,
                        per_row_fake_quant,
                        symmetric_quant
                    );
                } catch (...) {
                    // Some axis values may be invalid, continue testing
                }
            }
        }

        // Test with extreme averaging constants
        std::vector<double> test_constants = {0.001, 0.1, 0.5, 0.9, 0.999};
        for (double const_val : test_constants) {
            try {
                auto const_result = torch::fused_moving_avg_obs_fake_quant(
                    input, 
                    scale, 
                    zero_point, 
                    running_min, 
                    running_max, 
                    const_val,
                    quant_min,
                    quant_max,
                    ch_axis,
                    per_row_fake_quant,
                    symmetric_quant
                );
            } catch (...) {
                // Continue testing other values
            }
        }

        // Test with different quantization ranges
        std::vector<std::pair<int64_t, int64_t>> quant_ranges = {
            {-128, 127}, {0, 255}, {-32768, 32767}, {0, 65535}
        };
        for (auto range : quant_ranges) {
            try {
                auto range_result = torch::fused_moving_avg_obs_fake_quant(
                    input, 
                    scale, 
                    zero_point, 
                    running_min, 
                    running_max, 
                    averaging_constant,
                    range.first,
                    range.second,
                    ch_axis,
                    per_row_fake_quant,
                    symmetric_quant
                );
            } catch (...) {
                // Continue testing other ranges
            }
        }

        // Test with both per_row_fake_quant settings
        for (bool per_row : {true, false}) {
            for (bool symmetric : {true, false}) {
                try {
                    auto flag_result = torch::fused_moving_avg_obs_fake_quant(
                        input, 
                        scale, 
                        zero_point, 
                        running_min, 
                        running_max, 
                        averaging_constant,
                        quant_min,
                        quant_max,
                        ch_axis,
                        per_row,
                        symmetric
                    );
                } catch (...) {
                    // Continue testing other flag combinations
                }
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}