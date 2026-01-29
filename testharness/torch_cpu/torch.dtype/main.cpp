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
        
        // Create a tensor from the input data
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Test torch.dtype functionality
        auto dtype = tensor.dtype();
        
        // Test various ways to access dtype information
        auto scalar_type = dtype.toScalarType();
        
        // Test dtype equality comparison
        bool is_float = (dtype == torch::kFloat);
        bool is_double = (dtype == torch::kDouble);
        bool is_int = (dtype == torch::kInt);
        bool is_long = (dtype == torch::kLong);
        bool is_bool = (dtype == torch::kBool);
        bool is_half = (dtype == torch::kHalf);
        
        // Prevent compiler from optimizing away the comparisons
        (void)is_float;
        (void)is_double;
        (void)is_int;
        (void)is_long;
        (void)is_bool;
        (void)is_half;
        
        // Test dtype name and size
        std::string type_name = c10::toString(scalar_type);
        size_t element_size = dtype.itemsize();
        (void)element_size;
        
        // Test dtype conversion
        if (offset + 1 < Size) {
            uint8_t conversion_type = Data[offset++];
            torch::ScalarType target_type = fuzzer_utils::parseDataType(conversion_type);
            
            try {
                // Try to convert tensor to the target dtype
                // Some conversions may fail (e.g., complex to non-complex)
                torch::Tensor converted_tensor = tensor.to(target_type);
                
                // Verify the conversion worked
                auto new_dtype = converted_tensor.dtype();
                bool conversion_successful = (new_dtype.toScalarType() == target_type);
                (void)conversion_successful;
            } catch (const c10::Error&) {
                // Some dtype conversions are not supported, silently ignore
            }
        }
        
        // Test dtype properties using c10 utilities
        bool is_floating_point = c10::isFloatingType(scalar_type);
        bool is_complex = c10::isComplexType(scalar_type);
        bool is_integral = c10::isIntegralType(scalar_type, /*includeBool=*/true);
        (void)is_floating_point;
        (void)is_complex;
        (void)is_integral;
        
        // Test creating a new tensor with the same dtype
        if (offset + 4 < Size) {
            int64_t dim1 = (Data[offset++] % 4) + 1;
            int64_t dim2 = (Data[offset++] % 4) + 1;
            std::vector<int64_t> new_shape = {dim1, dim2};
            torch::Tensor new_tensor = torch::empty(new_shape, tensor.options());
            bool dtypes_match = (new_tensor.dtype() == tensor.dtype());
            (void)dtypes_match;
        }
        
        // Test dtype from scalar type conversion
        torch::Dtype dtype_from_scalar = torch::typeMetaToScalarType(dtype);
        (void)dtype_from_scalar;
        
        // Test zeros/ones with specific dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_choice = Data[offset++];
            torch::ScalarType chosen_type = fuzzer_utils::parseDataType(dtype_choice);
            
            try {
                auto opts = torch::TensorOptions().dtype(chosen_type);
                torch::Tensor zeros_tensor = torch::zeros({2, 2}, opts);
                torch::Tensor ones_tensor = torch::ones({2, 2}, opts);
                
                // Verify dtypes
                bool zeros_dtype_ok = (zeros_tensor.dtype().toScalarType() == chosen_type);
                bool ones_dtype_ok = (ones_tensor.dtype().toScalarType() == chosen_type);
                (void)zeros_dtype_ok;
                (void)ones_dtype_ok;
            } catch (const c10::Error&) {
                // Some dtypes may not be supported for certain operations
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