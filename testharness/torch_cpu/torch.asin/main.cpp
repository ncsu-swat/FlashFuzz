#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
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
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor for asin operation
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply asin operation
        torch::Tensor result = torch::asin(input);
        
        // Try in-place version if there's more data
        if (offset < Size) {
            torch::Tensor input_copy = input.clone();
            input_copy.asin_();
        }
        
        // Try with different options if there's more data
        if (offset + 1 < Size) {
            // Use the next byte to determine if we should try with different options
            uint8_t option_byte = Data[offset++];
            
            // Try with out tensor
            if (option_byte & 0x01) {
                torch::Tensor out = torch::empty_like(input);
                torch::asin_out(out, input);
            }
            
            // Try with different memory format if applicable
            if ((option_byte & 0x02) && input.dim() >= 4) {
                try {
                    torch::Tensor channels_last = input.to(torch::MemoryFormat::ChannelsLast);
                    torch::asin(channels_last);
                } catch (...) {
                    // Memory format conversion may fail for some tensor configurations
                }
            }
        }
        
        // Test with different dtypes to improve coverage
        if (offset < Size) {
            uint8_t dtype_byte = Data[offset++];
            try {
                if (dtype_byte & 0x01) {
                    torch::Tensor float_input = input.to(torch::kFloat32);
                    torch::asin(float_input);
                }
                if (dtype_byte & 0x02) {
                    torch::Tensor double_input = input.to(torch::kFloat64);
                    torch::asin(double_input);
                }
                if (dtype_byte & 0x04) {
                    // Complex input
                    torch::Tensor complex_input = input.to(torch::kComplexFloat);
                    torch::asin(complex_input);
                }
            } catch (...) {
                // Some dtype conversions may fail
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}