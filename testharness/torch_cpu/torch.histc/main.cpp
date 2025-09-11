#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract parameters for histc from the remaining data
        int64_t bins = 100;  // Default value
        double min = 0.0;    // Default value
        double max = 0.0;    // Default value
        
        // Parse bins parameter if we have enough data
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&bins, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure bins is positive (histc requires this)
            bins = std::abs(bins) % 1000 + 1;  // Limit to reasonable range
        }
        
        // Parse min parameter if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&min, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // Parse max parameter if we have enough data
        if (offset + sizeof(double) <= Size) {
            std::memcpy(&max, Data + offset, sizeof(double));
            offset += sizeof(double);
        }
        
        // If min > max, swap them
        if (min > max) {
            std::swap(min, max);
        }
        
        // If min == max, set max to min + 1 to avoid invalid range
        if (min == max) {
            max = min + 1.0;
        }
        
        // Apply histc operation
        torch::Tensor result;
        
        // Try different variants of the histc call
        if (offset % 3 == 0) {
            // Call with all parameters
            result = torch::histc(input, bins, min, max);
        } else if (offset % 3 == 1) {
            // Call with default min/max
            result = torch::histc(input, bins);
        } else {
            // Call with all defaults
            result = torch::histc(input);
        }
        
        // Ensure the result is valid
        if (result.numel() > 0) {
            // Access some elements to ensure computation happened
            auto sum = result.sum().item<double>();
            if (std::isnan(sum) || std::isinf(sum)) {
                throw std::runtime_error("Invalid result: NaN or Inf detected");
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
