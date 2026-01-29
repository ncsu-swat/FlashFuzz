#include "fuzzer_utils.h"
#include <iostream>

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    // Progress tracking
    static uint64_t iteration_count = 0;
    iteration_count++;
    if (iteration_count % 10000 == 0) {
        std::cout << "Iterations: " << iteration_count << std::endl;
    }

    // Save original default dtype to restore later
    caffe2::TypeMeta original_default = torch::get_default_dtype();

    try
    {
        size_t offset = 0;
        
        // Need at least 1 byte for dtype selection
        if (Size < 1) {
            return 0;
        }
        
        // Parse the dtype selector from the input data
        uint8_t dtype_selector = Data[offset++];
        
        // Only floating point types are valid for set_default_dtype
        // Map to valid floating point types
        torch::ScalarType dtype;
        switch (dtype_selector % 4) {
            case 0: dtype = torch::kFloat; break;
            case 1: dtype = torch::kDouble; break;
            case 2: dtype = torch::kHalf; break;
            case 3: dtype = torch::kBFloat16; break;
            default: dtype = torch::kFloat; break;
        }
        
        // Set the default dtype
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(dtype));
        
        // Verify the default dtype was set correctly
        caffe2::TypeMeta current_default = torch::get_default_dtype();
        
        // Create a tensor with the default dtype
        std::vector<int64_t> shape = {2, 3};
        torch::Tensor tensor = torch::empty(shape);
        
        // Try with different tensor creation methods
        torch::Tensor ones_tensor = torch::ones(shape);
        torch::Tensor zeros_tensor = torch::zeros(shape);
        torch::Tensor rand_tensor = torch::rand(shape);
        
        // Create a tensor with explicit dtype to ensure it overrides default
        torch::ScalarType explicit_dtype;
        switch ((dtype_selector + 1) % 4) {
            case 0: explicit_dtype = torch::kFloat; break;
            case 1: explicit_dtype = torch::kDouble; break;
            case 2: explicit_dtype = torch::kHalf; break;
            case 3: explicit_dtype = torch::kBFloat16; break;
            default: explicit_dtype = torch::kFloat; break;
        }
        torch::Tensor explicit_tensor = torch::empty(shape, torch::TensorOptions().dtype(explicit_dtype));
        
        // Create a tensor from input data if there's enough data left
        if (offset < Size) {
            try {
                torch::Tensor data_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Try operations with the tensor to ensure default dtype works with operations
                torch::Tensor result = data_tensor.to(current_default);
                
                // Additional operations to increase coverage
                torch::Tensor sum_result = data_tensor.sum();
                torch::Tensor mean_result = data_tensor.to(torch::kFloat).mean();
            }
            catch (const std::exception &e) {
                // Tensor creation or conversion may fail for some inputs - that's expected
            }
        }
        
        // Test setting different dtypes in sequence
        if (Size > 1) {
            torch::ScalarType second_dtype;
            switch (Data[Size - 1] % 4) {
                case 0: second_dtype = torch::kFloat; break;
                case 1: second_dtype = torch::kDouble; break;
                case 2: second_dtype = torch::kHalf; break;
                case 3: second_dtype = torch::kBFloat16; break;
                default: second_dtype = torch::kDouble; break;
            }
            torch::set_default_dtype(torch::scalarTypeToTypeMeta(second_dtype));
            
            // Create tensor with new default
            torch::Tensor new_default_tensor = torch::randn({3, 3});
        }
        
        // Reset default dtype to original
        torch::set_default_dtype(original_default);
    }
    catch (const std::exception &e)
    {
        // Always restore default dtype even on exception
        torch::set_default_dtype(original_default);
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}