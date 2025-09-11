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
        
        // Need at least a few bytes for basic tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract groups parameter from the input data
        int64_t groups = 1;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Ensure groups is not zero to avoid division by zero
            if (groups == 0) {
                groups = 1;
            }
        }
        
        // Try to apply channel_shuffle with different parameters
        try {
            // Apply channel_shuffle operation
            torch::Tensor output = torch::channel_shuffle(input, groups);
        } catch (...) {
            // Catch any exceptions from channel_shuffle and try with different groups
            try {
                // Try with a different groups value
                groups = 2;
                torch::Tensor output = torch::channel_shuffle(input, groups);
            } catch (...) {
                // Try with yet another groups value
                groups = 3;
                torch::Tensor output = torch::channel_shuffle(input, groups);
            }
        }
        
        // Try another approach with absolute value of groups
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Take absolute value and ensure it's at least 1
            groups = std::abs(groups);
            if (groups == 0) {
                groups = 1;
            }
            
            try {
                torch::Tensor output = torch::channel_shuffle(input, groups);
            } catch (...) {
                // Ignore exceptions
            }
        }
        
        // Try with a negative groups value to test error handling
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&groups, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
            
            // Make groups negative to test error handling
            if (groups > 0) {
                groups = -groups;
            } else if (groups == 0) {
                groups = -1;
            }
            
            try {
                torch::Tensor output = torch::channel_shuffle(input, groups);
            } catch (...) {
                // Ignore exceptions
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
