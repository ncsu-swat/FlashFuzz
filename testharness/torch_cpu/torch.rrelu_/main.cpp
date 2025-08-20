#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes for tensor creation
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Extract lower and upper bounds from remaining data
        float lower = 0.0;
        float upper = 0.0;
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&lower, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        if (offset + sizeof(float) <= Size) {
            std::memcpy(&upper, Data + offset, sizeof(float));
            offset += sizeof(float);
        }
        
        // Ensure lower <= upper (if not, swap them)
        if (lower > upper) {
            std::swap(lower, upper);
        }
        
        // Create a copy of the input tensor for testing
        torch::Tensor input_copy = input.clone();
        
        // Apply rrelu_ in-place
        if (offset < Size) {
            // Use the next byte to determine if we should provide a generator
            bool use_generator = (Data[offset] % 2 == 0);
            
            if (use_generator) {
                // Create a generator with a seed from the data
                uint64_t seed = 0;
                if (offset + sizeof(uint64_t) <= Size) {
                    std::memcpy(&seed, Data + offset, sizeof(uint64_t));
                    offset += sizeof(uint64_t);
                }
                
                auto generator = torch::make_generator<torch::CPUGeneratorImpl>(seed);
                torch::rrelu_(input, lower, upper, false, generator);
            } else {
                // Use the default generator
                torch::rrelu_(input, lower, upper);
            }
        } else {
            // Use default parameters
            torch::rrelu_(input);
        }
        
        // Verify that the operation was applied in-place
        if (input.data_ptr() != input_copy.data_ptr()) {
            throw std::runtime_error("rrelu_ should modify the tensor in-place");
        }
        
        // Test non-inplace version for comparison
        torch::Tensor output = torch::rrelu(input_copy, lower, upper);
        
        // Try to access elements to check for potential crashes
        if (input.numel() > 0) {
            input.item();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}