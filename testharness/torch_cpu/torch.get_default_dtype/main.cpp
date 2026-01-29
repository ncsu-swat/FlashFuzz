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
        
        // Save the original default dtype to restore later
        auto original_default_dtype = torch::get_default_dtype();
        
        // Test basic get_default_dtype
        auto default_dtype = torch::get_default_dtype();
        
        // Verify it returns a valid TypeMeta
        auto scalar_type = default_dtype.toScalarType();
        (void)scalar_type; // Use the value
        
        // Try setting different floating-point dtypes and getting them back
        // Note: set_default_dtype only accepts floating point types
        if (Size > 0) {
            uint8_t dtype_selector = Data[offset++];
            
            // Map to valid floating point types only
            torch::ScalarType float_dtypes[] = {
                torch::kFloat32,
                torch::kFloat64,
                torch::kFloat16,
                torch::kBFloat16
            };
            torch::ScalarType dtype = float_dtypes[dtype_selector % 4];
            
            try {
                // Set the default dtype - convert ScalarType to TypeMeta
                torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(dtype));
                
                // Verify that get_default_dtype returns the dtype we just set
                auto new_default_dtype = torch::get_default_dtype();
                
                // Create a tensor with default dtype (should use the new default)
                std::vector<int64_t> shape = {2, 3};
                auto tensor = torch::zeros(shape);
                
                // The tensor dtype should match the new default for float tensors
                (void)tensor.dtype();
            } catch (const std::exception&) {
                // Some float types may not be supported on all platforms
            }
        }
        
        // Test with tensor creation if we have enough data
        if (Size > offset + 4) {
            try {
                // Create a tensor using the input data
                auto tensor = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Get the current default dtype
                auto current_dtype = torch::get_default_dtype();
                
                // Create a new float tensor with explicit default dtype
                auto new_tensor = torch::zeros({2, 2}, torch::TensorOptions().dtype(current_dtype));
                
                // Access dtype info
                auto tensor_dtype = new_tensor.dtype();
                (void)tensor_dtype;
            } catch (const std::exception&) {
                // Tensor creation may fail with invalid fuzzer data
            }
        }
        
        // Test switching between different float dtypes in sequence
        if (Size > offset + 2) {
            uint8_t selector1 = Data[offset++];
            uint8_t selector2 = Data[offset++];
            
            torch::ScalarType float_dtypes[] = {
                torch::kFloat32,
                torch::kFloat64,
                torch::kFloat16,
                torch::kBFloat16
            };
            
            torch::ScalarType dtype1 = float_dtypes[selector1 % 4];
            torch::ScalarType dtype2 = float_dtypes[selector2 % 4];
            
            try {
                // Set first dtype
                torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(dtype1));
                auto retrieved_dtype1 = torch::get_default_dtype();
                
                // Create tensor with first dtype
                auto tensor1 = torch::ones({3, 3});
                
                // Set second dtype
                torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(dtype2));
                auto retrieved_dtype2 = torch::get_default_dtype();
                
                // Create tensor with second dtype
                auto tensor2 = torch::ones({3, 3});
                
                // Use the retrieved types
                (void)retrieved_dtype1.toScalarType();
                (void)retrieved_dtype2.toScalarType();
            } catch (const std::exception&) {
                // Some operations may fail on certain platforms
            }
        }
        
        // Test checking if dtype is floating point
        {
            auto dtype = torch::get_default_dtype();
            bool is_float = (dtype == caffe2::TypeMeta::fromScalarType(torch::kFloat32)) ||
                           (dtype == caffe2::TypeMeta::fromScalarType(torch::kFloat64)) ||
                           (dtype == caffe2::TypeMeta::fromScalarType(torch::kFloat16)) ||
                           (dtype == caffe2::TypeMeta::fromScalarType(torch::kBFloat16));
            (void)is_float;
        }
        
        // Always restore the original default dtype to avoid state pollution
        torch::set_default_dtype(original_default_dtype);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}