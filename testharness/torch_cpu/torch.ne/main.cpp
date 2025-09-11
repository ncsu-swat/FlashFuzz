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
            
            // Test tensor != scalar
            torch::Tensor result = tensor1.ne(scalar_value);
        } else {
            // Create a second tensor for comparison
            if (offset < Size) {
                tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                
                // Test tensor != tensor
                torch::Tensor result = tensor1.ne(tensor2);
            } else {
                // Not enough data for second tensor, use a simple value
                tensor2 = torch::tensor(1);
                torch::Tensor result = tensor1.ne(tensor2);
            }
        }
        
        // Test the out= variant if we have both tensors
        if (!use_scalar && tensor2.defined()) {
            // Try to match shapes if possible for out= variant
            torch::Tensor out;
            try {
                // Broadcasting rules apply, so we need a shape that can hold the result
                std::vector<int64_t> out_shape;
                if (tensor1.dim() >= tensor2.dim()) {
                    out_shape = tensor1.sizes().vec();
                } else {
                    out_shape = tensor2.sizes().vec();
                }
                
                out = torch::empty(out_shape, torch::kBool);
                torch::ne_out(out, tensor1, tensor2);
            } catch (const std::exception& e) {
                // Broadcasting might fail, that's expected in some cases
            }
        }
        
        // Test the static function variant
        if (!use_scalar && tensor2.defined()) {
            torch::Tensor result = torch::ne(tensor1, tensor2);
        } else if (use_scalar) {
            torch::Tensor result = torch::ne(tensor1, scalar_value);
        }
        
        // Test the operator!= variant
        if (!use_scalar && tensor2.defined()) {
            torch::Tensor result = tensor1 != tensor2;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
