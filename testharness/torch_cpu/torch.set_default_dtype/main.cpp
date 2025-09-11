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
        
        // Need at least 1 byte for dtype selection
        if (Size < 1) {
            return 0;
        }
        
        // Parse the dtype selector from the input data
        uint8_t dtype_selector = Data[offset++];
        torch::ScalarType dtype = fuzzer_utils::parseDataType(dtype_selector);
        
        // Set the default dtype
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(dtype));
        
        // Verify the default dtype was set correctly
        caffe2::TypeMeta current_default = torch::get_default_dtype();
        
        // Create a tensor with the default dtype
        std::vector<int64_t> shape = {2, 3};
        torch::Tensor tensor = torch::empty(shape);
        
        // Check if the tensor has the expected dtype
        if (tensor.dtype() != current_default) {
            throw std::runtime_error("Tensor dtype doesn't match the default dtype");
        }
        
        // Try with different tensor creation methods
        torch::Tensor ones_tensor = torch::ones(shape);
        torch::Tensor zeros_tensor = torch::zeros(shape);
        torch::Tensor rand_tensor = torch::rand(shape);
        
        // Create a tensor with explicit dtype to ensure it overrides default
        torch::ScalarType explicit_dtype = fuzzer_utils::parseDataType((dtype_selector + 1) % 255);
        torch::Tensor explicit_tensor = torch::empty(shape, torch::TensorOptions().dtype(explicit_dtype));
        
        // Check if explicit dtype is respected
        if (explicit_tensor.dtype() != torch::scalarTypeToTypeMeta(explicit_dtype)) {
            throw std::runtime_error("Explicit dtype not respected");
        }
        
        // Create a tensor from input data if there's enough data left
        if (offset < Size) {
            torch::Tensor data_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try operations with the tensor to ensure default dtype works with operations
            torch::Tensor result = data_tensor.to(current_default);
        }
        
        // Reset default dtype to float (standard default)
        torch::set_default_dtype(torch::scalarTypeToTypeMeta(torch::kFloat));
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
