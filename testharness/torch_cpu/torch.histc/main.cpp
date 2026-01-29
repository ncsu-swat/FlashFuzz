#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <cmath>          // For std::isnan, std::isinf

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // histc requires floating-point input
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Extract parameters for histc from the remaining data
        int64_t bins = 100;  // Default value
        double min_val = 0.0;    // Default value
        double max_val = 0.0;    // Default value
        
        // Parse bins parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&bins, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure bins is positive (histc requires this)
            bins = std::abs(bins) % 1000 + 1;  // Limit to reasonable range
        }
        
        // Parse min parameter if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Handle NaN/Inf
            if (std::isnan(min_val) || std::isinf(min_val)) {
                min_val = 0.0;
            }
        }
        
        // Parse max parameter if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max_val, Data + offset, sizeof(double));
            offset += sizeof(double);
            
            // Handle NaN/Inf
            if (std::isnan(max_val) || std::isinf(max_val)) {
                max_val = 0.0;
            }
        }
        
        // If min > max, swap them
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // If min == max, set max to min + 1 to avoid invalid range
        if (min_val == max_val) {
            max_val = min_val + 1.0;
        }
        
        // Apply histc operation with different variants
        torch::Tensor result;
        
        // Use a byte from data to select variant if available
        uint8_t variant = (offset < Size) ? Data[offset] % 3 : 0;
        
        if (variant == 0) {
            // Call with all parameters
            result = torch::histc(input, bins, min_val, max_val);
        } else if (variant == 1) {
            // Call with default min/max (auto-computed from input range)
            result = torch::histc(input, bins);
        } else {
            // Call with default bins (100) and default min/max
            result = torch::histc(input);
        }
        
        // Access result to ensure computation completed
        (void)result.numel();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}