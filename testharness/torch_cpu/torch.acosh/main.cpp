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
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply acosh operation
        torch::Tensor result = torch::acosh(input);
        
        // Try in-place version
        if (offset < Size) {
            try {
                // Clone and convert to float for in-place operation (requires floating point)
                torch::Tensor input_copy = input.to(torch::kFloat).clone();
                input_copy.acosh_();
            } catch (const std::exception &) {
                // In-place may fail for certain dtypes, silently ignore
            }
        }
        
        // Try with different dtypes if there's more data
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset++];
            
            try {
                // acosh requires floating point types
                torch::ScalarType dtype;
                switch (dtype_selector % 4) {
                    case 0: dtype = torch::kFloat; break;
                    case 1: dtype = torch::kDouble; break;
                    case 2: dtype = torch::kHalf; break;
                    case 3: dtype = torch::kBFloat16; break;
                    default: dtype = torch::kFloat; break;
                }
                
                torch::Tensor float_input = input.to(dtype);
                torch::Tensor result_with_dtype = torch::acosh(float_input);
                
                // Try acosh_out - output must have same dtype as input
                torch::Tensor output = torch::empty_like(float_input);
                torch::acosh_out(output, float_input);
            } catch (const std::exception &) {
                // dtype conversion may fail, silently ignore
            }
        }
        
        // Try with different memory formats if there's more data
        if (offset < Size) {
            uint8_t format_selector = Data[offset++];
            
            try {
                // Try different memory formats
                if (format_selector % 3 == 0 && input.dim() >= 4) {
                    auto channels_last_input = input.to(torch::MemoryFormat::ChannelsLast);
                    torch::Tensor channels_last_result = torch::acosh(channels_last_input);
                } else if (format_selector % 3 == 1 && input.dim() >= 5) {
                    auto channels_last_3d_input = input.to(torch::MemoryFormat::ChannelsLast3d);
                    torch::Tensor channels_last_3d_result = torch::acosh(channels_last_3d_input);
                }
            } catch (const std::exception &) {
                // Memory format conversion may fail, silently ignore
            }
        }
        
        // Try with non-contiguous tensor if there's more data
        if (offset < Size && input.dim() > 0 && input.numel() > 1) {
            try {
                // Create a non-contiguous view if possible
                if (input.dim() > 1 && input.size(0) > 1) {
                    torch::Tensor non_contiguous = input.slice(0, 0, input.size(0), 2);
                    if (!non_contiguous.is_contiguous()) {
                        torch::Tensor non_contiguous_result = torch::acosh(non_contiguous);
                    }
                }
            } catch (const std::exception &) {
                // Non-contiguous operations may fail, silently ignore
            }
        }
        
        // Try with complex tensors if there's more data
        if (offset < Size) {
            try {
                torch::Tensor complex_input = input.to(torch::kComplexFloat);
                torch::Tensor complex_result = torch::acosh(complex_input);
            } catch (const std::exception &) {
                // Complex conversion may fail, silently ignore
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}