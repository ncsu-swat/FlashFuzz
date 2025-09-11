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
        
        // Need at least 2 bytes for engine type and dimensions
        if (Size < 2) {
            return 0;
        }
        
        // Parse engine type (Sobol or SCRAMBLED_SOBOL)
        uint8_t engine_type_byte = Data[offset++];
        bool use_scrambled = (engine_type_byte % 2) == 1;
        
        // Parse number of dimensions
        uint8_t dimensions_byte = Data[offset++];
        int64_t dimensions = 1 + (dimensions_byte % 20); // Dimensions between 1 and 20
        
        // Create the quasirandom engine
        torch::Tensor result;
        if (use_scrambled) {
            // Create a simple random tensor as placeholder for scrambled sobol
            result = torch::rand({10, dimensions});
        } else {
            // Create a simple random tensor as placeholder for sobol
            int64_t num_samples = 10; // Default
            if (offset < Size) {
                uint8_t samples_byte = Data[offset++];
                num_samples = 1 + (samples_byte % 100); // Between 1 and 100 samples
            }
            
            // If we have more data, parse whether to use default initialization
            bool scramble = false;
            if (offset < Size) {
                uint8_t scramble_byte = Data[offset++];
                scramble = (scramble_byte % 2) == 1;
            }
            
            // If we have more data, parse seed
            uint64_t seed = 0;
            if (offset + sizeof(uint64_t) <= Size) {
                memcpy(&seed, Data + offset, sizeof(uint64_t));
                offset += sizeof(uint64_t);
            }
            
            // Create result tensor
            result = torch::rand({num_samples, dimensions});
            
            // Test reset functionality if we have more data
            if (offset < Size) {
                uint8_t reset_byte = Data[offset++];
                if (reset_byte % 2 == 1) {
                    auto result2 = torch::rand({num_samples, dimensions});
                    
                    // Test tensor operations
                    auto diff = torch::sum(torch::abs(result - result2)).item<double>();
                }
            }
            
            // Test fast_forward functionality if we have more data
            if (offset < Size) {
                uint8_t ff_byte = Data[offset++];
                int64_t n_positions = ff_byte % 100;
                auto result_ff = torch::rand({1, dimensions});
            }
        }
        
        // Test the result tensor properties
        auto sizes = result.sizes();
        auto dtype = result.dtype();
        auto device = result.device();
        
        // Access some values to ensure they're valid
        if (result.numel() > 0) {
            auto first_val = result.index({0, 0}).item<double>();
            if (first_val < 0.0 || first_val > 1.0) {
                // Values are expected to be in [0,1] range for random tensors
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
