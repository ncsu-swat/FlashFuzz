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
        
        // Need at least some data to create tensors
        if (Size < 4) {
            return 0;
        }
        
        // Create first tensor
        torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create second tensor or use a scalar
        bool use_scalar = false;
        torch::Tensor tensor2;
        torch::Scalar scalar_value;
        
        // Use remaining bytes to decide whether to use a scalar or tensor
        if (offset < Size) {
            use_scalar = Data[offset++] % 2 == 0;
        }
        
        if (use_scalar && offset < Size) {
            // Create a scalar value from the next bytes
            int64_t scalar_int = 0;
            size_t bytes_to_copy = std::min(sizeof(int64_t), Size - offset);
            std::memcpy(&scalar_int, Data + offset, bytes_to_copy);
            offset += bytes_to_copy;
            scalar_value = torch::Scalar(scalar_int);
            
            // Test tensor != scalar (method variant)
            torch::Tensor result = tensor1.ne(scalar_value);
            
            // Test the static function variant with scalar
            torch::Tensor result2 = torch::ne(tensor1, scalar_value);
        } else {
            // Create a second tensor for comparison
            if (offset < Size) {
                tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
            } else {
                // Not enough data for second tensor, use a simple value
                tensor2 = torch::tensor(1);
            }
            
            // Test tensor != tensor (method variant)
            torch::Tensor result = tensor1.ne(tensor2);
            
            // Test the static function variant
            torch::Tensor result2 = torch::ne(tensor1, tensor2);
            
            // Test the operator!= variant
            torch::Tensor result3 = tensor1 != tensor2;
            
            // Test the out= variant
            try {
                // Use at::infer_size to compute broadcasted shape
                auto broadcasted_sizes = at::infer_size(tensor1.sizes(), tensor2.sizes());
                torch::Tensor out = torch::empty(broadcasted_sizes, torch::kBool);
                torch::ne_out(out, tensor1, tensor2);
            } catch (...) {
                // Broadcasting might fail, that's expected in some cases
            }
        }
        
        // Test ne_ (in-place) variant - only works on bool tensors
        try {
            torch::Tensor bool_tensor = tensor1.to(torch::kBool).clone();
            if (use_scalar) {
                bool_tensor.ne_(scalar_value);
            } else if (tensor2.defined()) {
                bool_tensor.ne_(tensor2);
            }
        } catch (...) {
            // In-place comparison might fail due to type/shape constraints
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0;  // Keep the input
}