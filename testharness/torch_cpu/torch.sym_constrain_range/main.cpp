#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for the input tensor and range parameters
        if (Size < 8) {
            return 0;
        }
        
        // Extract min and max values from the data
        int64_t min_val = 0;
        int64_t max_val = 100;
        
        if (offset + sizeof(int64_t) * 2 <= Size) {
            std::memcpy(&min_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            std::memcpy(&max_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Swap if min > max to ensure valid range
        if (min_val > max_val) {
            std::swap(min_val, max_val);
        }
        
        // Create a scalar value from remaining data
        int64_t scalar_val = min_val;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&scalar_val, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Apply sym_constrain_range operation with scalar input
        torch::sym_constrain_range(at::Scalar(scalar_val), min_val, max_val);
        
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}