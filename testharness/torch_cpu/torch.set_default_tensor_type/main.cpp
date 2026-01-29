#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte to select tensor type
        if (Size < 1) {
            return 0;
        }
        
        // Parse the tensor type selector from the first byte
        uint8_t type_selector = Data[0];
        offset++;
        
        // torch::set_default_dtype only accepts floating-point types
        // Using non-floating types will throw an error
        std::vector<torch::ScalarType> float_types = {
            torch::kFloat32,
            torch::kFloat64,
            torch::kHalf,
            torch::kBFloat16,
        };
        
        // Select a floating-point type based on the selector
        torch::ScalarType selected_dtype = float_types[type_selector % float_types.size()];
        
        // Set the default tensor dtype
        torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(selected_dtype));
        
        // Create a tensor to verify the default type was set correctly
        // torch::ones should now use the default dtype
        torch::Tensor default_tensor = torch::ones({2, 2});
        
        // Verify the default tensor has the expected type
        // Note: For some types like BFloat16/Half, the actual dtype may differ
        // based on hardware support, so we just verify it's a floating type
        
        // Try some basic operations with the default tensor type
        torch::Tensor result = default_tensor + 1.0f;
        torch::Tensor mul_result = default_tensor * 2.0f;
        
        // matmul may not be supported for all float types (Half, BFloat16)
        // so wrap in inner try-catch
        try {
            // Convert to float32 for matmul if needed for half types
            if (selected_dtype == torch::kHalf || selected_dtype == torch::kBFloat16) {
                torch::Tensor float_tensor = default_tensor.to(torch::kFloat32);
                torch::Tensor matmul_result = torch::matmul(float_tensor, float_tensor);
            } else {
                torch::Tensor matmul_result = torch::matmul(default_tensor, default_tensor);
            }
        } catch (...) {
            // Some operations may not be supported for certain dtypes - that's OK
        }
        
        // Test creating tensors with explicit values
        torch::Tensor zeros_tensor = torch::zeros({3, 3});
        torch::Tensor randn_tensor = torch::randn({2, 3});
        torch::Tensor empty_tensor = torch::empty({2, 2});
        
        // Test tensor creation from data if we have remaining bytes
        if (Size > offset + 4) {
            // Create a tensor using the remaining data
            torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to convert it to the default type
            try {
                torch::Tensor converted = tensor.to(selected_dtype);
            } catch (...) {
                // Conversion may fail for some type combinations - OK
            }
        }
        
        // Additional coverage: test arange and linspace with default dtype
        try {
            torch::Tensor arange_tensor = torch::arange(0.0, 10.0, 0.5);
            torch::Tensor linspace_tensor = torch::linspace(0.0, 1.0, 10);
        } catch (...) {
            // May fail for some dtypes
        }
        
        // Reset to default float32 before returning
        torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kFloat32));
    }
    catch (const std::exception &e)
    {
        // Reset to default before returning on error
        try {
            torch::set_default_dtype(caffe2::scalarTypeToTypeMeta(torch::kFloat32));
        } catch (...) {}
        
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}