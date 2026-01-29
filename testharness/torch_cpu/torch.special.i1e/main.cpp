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
        if (Size < 2) {
            return 0;
        }
        
        size_t offset = 0;
        
        // Create input tensor for torch.special.i1e
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure we have a floating point tensor (i1e requires float types)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Apply torch.special.i1e operation
        torch::Tensor result = torch::special::i1e(input);
        
        // Force computation by accessing result
        if (result.defined() && result.numel() > 0) {
            volatile float sum = result.sum().item<float>();
            (void)sum;
        }
        
        // Test with output tensor variant if we have more data
        if (offset < Size) {
            torch::Tensor output = torch::empty_like(input);
            torch::special::i1e_out(output, input);
            
            if (output.defined() && output.numel() > 0) {
                volatile float sum = output.sum().item<float>();
                (void)sum;
            }
        }
        
        // Test with double precision based on fuzzer data
        if (offset < Size && (Data[offset % Size] & 0x01)) {
            torch::Tensor input_double = input.to(torch::kFloat64);
            torch::Tensor result_double = torch::special::i1e(input_double);
            
            if (result_double.defined() && result_double.numel() > 0) {
                volatile double sum = result_double.sum().item<double>();
                (void)sum;
            }
        }
        
        // Test with scalar input based on fuzzer data
        if (offset + 4 <= Size) {
            float scalar_val;
            std::memcpy(&scalar_val, Data + offset, sizeof(float));
            offset += sizeof(float);
            
            // Handle potential NaN/Inf from raw bytes - i1e handles these gracefully
            torch::Tensor scalar_tensor = torch::tensor(scalar_val);
            torch::Tensor scalar_result = torch::special::i1e(scalar_tensor);
            
            if (scalar_result.defined()) {
                volatile float val = scalar_result.item<float>();
                (void)val;
            }
        }
        
        // Test with different shapes occasionally (controlled by fuzzer data)
        if (offset < Size && (Data[offset % Size] & 0x02)) {
            try {
                // Create a multi-dimensional tensor
                int dim1 = (Data[offset % Size] % 8) + 1;
                int dim2 = ((offset + 1 < Size ? Data[(offset + 1) % Size] : 1) % 8) + 1;
                torch::Tensor multi_dim = torch::randn({dim1, dim2});
                torch::Tensor multi_result = torch::special::i1e(multi_dim);
                
                if (multi_result.defined() && multi_result.numel() > 0) {
                    volatile float sum = multi_result.sum().item<float>();
                    (void)sum;
                }
            } catch (...) {
                // Silently ignore shape-related issues
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