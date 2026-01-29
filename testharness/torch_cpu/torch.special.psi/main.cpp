#include "fuzzer_utils.h"
#include <iostream>

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
        
        if (Size < 4) {
            return 0;
        }
        
        // Read control byte for dtype selection
        uint8_t dtype_selector = Data[offset++];
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // psi requires floating point input
        torch::ScalarType dtype;
        switch (dtype_selector % 3) {
            case 0:
                dtype = torch::kFloat32;
                break;
            case 1:
                dtype = torch::kFloat64;
                break;
            default:
                dtype = torch::kFloat32;
                break;
        }
        input = input.to(dtype);
        
        // Apply torch.special.psi operation (digamma function)
        torch::Tensor result = torch::special::psi(input);
        
        // Force computation
        result.sum().item<double>();
        
        // Test the out variant
        if (offset < Size) {
            uint8_t test_out = Data[offset++];
            if (test_out % 2 == 0) {
                try {
                    torch::Tensor out = torch::empty_like(input);
                    torch::special::psi_out(out, input);
                    out.sum().item<double>();
                } catch (const std::exception &) {
                    // Silent catch for expected errors in out variant
                }
            }
        }
        
        // Test with different shaped tensors
        if (offset + 2 < Size) {
            uint8_t dim1 = Data[offset++] % 8 + 1;
            uint8_t dim2 = Data[offset++] % 8 + 1;
            
            try {
                torch::Tensor shaped_input = torch::rand({dim1, dim2}, torch::dtype(dtype));
                torch::Tensor shaped_result = torch::special::psi(shaped_input);
                shaped_result.sum().item<double>();
            } catch (const std::exception &) {
                // Silent catch for shape-related errors
            }
        }
        
        // Test with scalar input
        if (offset < Size) {
            try {
                double scalar_val = static_cast<double>(Data[offset++]) / 25.5 + 0.1; // Avoid 0 and negative integers
                torch::Tensor scalar_input = torch::tensor(scalar_val, torch::dtype(dtype));
                torch::Tensor scalar_result = torch::special::psi(scalar_input);
                scalar_result.item<double>();
            } catch (const std::exception &) {
                // Silent catch
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}