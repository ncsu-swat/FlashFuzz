#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some bytes for parameters
        if (Size < 16) {
            return 0;
        }

        // Extract tensor parameters
        auto tensor_params = extract_tensor_params(Data, Size, offset);
        if (tensor_params.empty()) {
            return 0;
        }

        // Create input tensor with various data types and shapes
        torch::Tensor input;
        uint8_t tensor_type = Data[offset++] % 4;
        
        switch (tensor_type) {
            case 0: {
                // Float tensor
                input = create_tensor<float>(tensor_params[0]);
                break;
            }
            case 1: {
                // Double tensor
                input = create_tensor<double>(tensor_params[0]);
                break;
            }
            case 2: {
                // Integer tensor
                input = create_tensor<int32_t>(tensor_params[0]);
                break;
            }
            case 3: {
                // Long tensor
                input = create_tensor<int64_t>(tensor_params[0]);
                break;
            }
        }

        if (input.numel() == 0) {
            return 0;
        }

        // Extract bins parameter (1 to 1000 to avoid excessive memory usage)
        int bins = 1;
        if (offset < Size) {
            bins = std::max(1, std::min(1000, static_cast<int>(Data[offset++]) + 1));
        }

        // Extract min and max parameters
        double min_val = 0.0;
        double max_val = 0.0;
        
        if (offset + 8 <= Size) {
            // Use bytes to create min value
            uint64_t min_bits = 0;
            for (int i = 0; i < 8 && offset < Size; i++) {
                min_bits |= (static_cast<uint64_t>(Data[offset++]) << (i * 8));
            }
            min_val = *reinterpret_cast<double*>(&min_bits);
            
            // Handle NaN and infinity
            if (std::isnan(min_val) || std::isinf(min_val)) {
                min_val = 0.0;
            }
        }

        if (offset + 8 <= Size) {
            // Use bytes to create max value
            uint64_t max_bits = 0;
            for (int i = 0; i < 8 && offset < Size; i++) {
                max_bits |= (static_cast<uint64_t>(Data[offset++]) << (i * 8));
            }
            max_val = *reinterpret_cast<double*>(&max_bits);
            
            // Handle NaN and infinity
            if (std::isnan(max_val) || std::isinf(max_val)) {
                max_val = 0.0;
            }
        }

        // Ensure min <= max, swap if necessary
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }

        // Test different scenarios based on remaining data
        uint8_t test_case = 0;
        if (offset < Size) {
            test_case = Data[offset++] % 6;
        }

        torch::Tensor result;

        switch (test_case) {
            case 0: {
                // Basic histc call
                result = torch::histc(input, bins, min_val, max_val);
                break;
            }
            case 1: {
                // histc with min=max=0 (auto range)
                result = torch::histc(input, bins, 0, 0);
                break;
            }
            case 2: {
                // histc with only bins specified
                result = torch::histc(input, bins);
                break;
            }
            case 3: {
                // histc with output tensor
                torch::Tensor out = torch::empty({bins}, input.options().dtype(torch::kFloat));
                result = torch::histc(input, bins, min_val, max_val, out);
                break;
            }
            case 4: {
                // Test with very small range
                double small_range = 1e-10;
                result = torch::histc(input, bins, min_val, min_val + small_range);
                break;
            }
            case 5: {
                // Test with tensor containing special values
                torch::Tensor special_input = input.clone();
                if (special_input.dtype().isFloatingPoint() && special_input.numel() > 0) {
                    // Add some NaN and inf values
                    auto flat = special_input.flatten();
                    if (flat.numel() > 0) {
                        flat[0] = std::numeric_limits<double>::quiet_NaN();
                    }
                    if (flat.numel() > 1) {
                        flat[1] = std::numeric_limits<double>::infinity();
                    }
                    if (flat.numel() > 2) {
                        flat[2] = -std::numeric_limits<double>::infinity();
                    }
                }
                result = torch::histc(special_input, bins, min_val, max_val);
                break;
            }
        }

        // Verify result properties
        if (result.defined()) {
            // Check that result has correct shape
            if (result.dim() != 1 || result.size(0) != bins) {
                std::cerr << "Unexpected result shape" << std::endl;
            }
            
            // Check that result is non-negative (histogram counts)
            if (result.dtype().isFloatingPoint()) {
                auto min_val_result = torch::min(result);
                if (min_val_result.item<double>() < 0) {
                    std::cerr << "Negative histogram count detected" << std::endl;
                }
            }
            
            // Check that sum of histogram equals number of valid elements
            // (excluding NaN and out-of-range values)
            auto sum_result = torch::sum(result);
            if (sum_result.dtype().isFloatingPoint()) {
                double total_count = sum_result.item<double>();
                if (total_count > input.numel()) {
                    std::cerr << "Histogram count exceeds input size" << std::endl;
                }
            }
        }

        // Test edge cases with different bin counts
        if (offset < Size) {
            uint8_t edge_case = Data[offset++] % 4;
            switch (edge_case) {
                case 0: {
                    // Single bin
                    auto single_bin_result = torch::histc(input, 1, min_val, max_val);
                    break;
                }
                case 1: {
                    // Many bins (but reasonable)
                    auto many_bins_result = torch::histc(input, 500, min_val, max_val);
                    break;
                }
                case 2: {
                    // Test with empty tensor
                    auto empty_input = torch::empty({0}, input.options());
                    auto empty_result = torch::histc(empty_input, bins, min_val, max_val);
                    break;
                }
                case 3: {
                    // Test with 1D tensor of different sizes
                    if (input.numel() > 0) {
                        auto reshaped = input.flatten();
                        auto reshaped_result = torch::histc(reshaped, bins, min_val, max_val);
                    }
                    break;
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