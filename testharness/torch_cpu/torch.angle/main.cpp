#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr/cout

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
        if (Size < 4) {
            return 0;
        }
        
        // Use first byte to select dtype variant
        uint8_t dtype_selector = Data[offset++];
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // torch::angle computes the element-wise angle (in radians) of a complex tensor
        // For real-valued tensors, it returns 0 for positive values and pi for negative values
        
        // Apply torch.angle operation on the created tensor
        torch::Tensor result = torch::angle(input_tensor);
        
        // Test with different dtype conversions based on fuzzer input
        if (offset < Size) {
            uint8_t variant = Data[offset++];
            
            try {
                if (variant % 4 == 0) {
                    // Test with complex float tensor
                    torch::Tensor complex_tensor = input_tensor.to(torch::kComplexFloat);
                    torch::Tensor complex_result = torch::angle(complex_tensor);
                }
                else if (variant % 4 == 1) {
                    // Test with complex double tensor
                    torch::Tensor complex_tensor = input_tensor.to(torch::kComplexDouble);
                    torch::Tensor complex_result = torch::angle(complex_tensor);
                }
                else if (variant % 4 == 2) {
                    // Test with float tensor (real-valued)
                    torch::Tensor float_tensor = input_tensor.to(torch::kFloat);
                    torch::Tensor float_result = torch::angle(float_tensor);
                }
                else {
                    // Test with double tensor (real-valued)
                    torch::Tensor double_tensor = input_tensor.to(torch::kDouble);
                    torch::Tensor double_result = torch::angle(double_tensor);
                }
            }
            catch (const std::exception &) {
                // Silently catch dtype conversion errors (expected for some inputs)
            }
        }
        
        // Test out variant if available - use torch::angle_outf or direct assignment
        if (offset < Size) {
            uint8_t out_variant = Data[offset++];
            
            if (out_variant % 2 == 0) {
                try {
                    // Create output tensor and use out variant
                    // torch::angle doesn't have a direct _out variant in all versions,
                    // so we test by assigning to pre-allocated tensor
                    torch::Tensor out_tensor = torch::empty_like(input_tensor);
                    out_tensor = torch::angle(input_tensor);
                }
                catch (const std::exception &) {
                    // Silently catch errors from out variant attempts
                }
            }
        }
        
        // Test with different tensor shapes
        if (offset + 2 < Size) {
            try {
                uint8_t dim1 = (Data[offset++] % 8) + 1;  // 1-8
                uint8_t dim2 = (Data[offset++] % 8) + 1;  // 1-8
                
                // Create a specific shaped tensor
                torch::Tensor shaped_tensor = torch::randn({dim1, dim2});
                torch::Tensor shaped_result = torch::angle(shaped_tensor);
                
                // Also try with complex values
                torch::Tensor complex_shaped = torch::randn({dim1, dim2}, torch::kComplexFloat);
                torch::Tensor complex_shaped_result = torch::angle(complex_shaped);
            }
            catch (const std::exception &) {
                // Silently catch shape-related errors
            }
        }
        
        // Test with scalar tensor
        if (offset < Size) {
            try {
                torch::Tensor scalar_tensor = torch::tensor(static_cast<float>(Data[offset++]) - 128.0f);
                torch::Tensor scalar_result = torch::angle(scalar_tensor);
            }
            catch (const std::exception &) {
                // Silently catch scalar conversion errors
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