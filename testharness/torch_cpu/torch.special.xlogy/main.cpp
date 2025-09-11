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
        
        // Create input tensors for torch.special.xlogy
        torch::Tensor x = fuzzer_utils::createTensor(Data, Size, offset);
        torch::Tensor y = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Try different variants of the xlogy operation
        
        // 1. Call xlogy with two tensors
        torch::Tensor result1 = torch::special::xlogy(x, y);
        
        // 2. Call xlogy with scalar and tensor
        if (Size > offset) {
            double scalar_value = static_cast<double>(Data[offset++]);
            torch::Tensor result2 = torch::special::xlogy(scalar_value, y);
        }
        
        // 3. Call xlogy with tensor and scalar
        if (Size > offset) {
            double scalar_value = static_cast<double>(Data[offset++]);
            torch::Tensor result3 = torch::special::xlogy(x, scalar_value);
        }
        
        // 4. Call xlogy with out parameter
        torch::Tensor out = torch::empty_like(x);
        torch::special::xlogy_out(out, x, y);
        
        // 5. Try with different dtypes if possible
        if (Size > offset + 2) {
            auto dtype_selector = Data[offset++] % 3;
            torch::ScalarType dtype;
            
            switch (dtype_selector) {
                case 0:
                    dtype = torch::kFloat;
                    break;
                case 1:
                    dtype = torch::kDouble;
                    break;
                case 2:
                    dtype = torch::kHalf;
                    break;
                default:
                    dtype = torch::kFloat;
            }
            
            // Convert tensors to the selected dtype
            torch::Tensor x_converted = x.to(dtype);
            torch::Tensor y_converted = y.to(dtype);
            
            // Call xlogy with converted tensors
            torch::Tensor result_converted = torch::special::xlogy(x_converted, y_converted);
        }
        
        // 6. Try with broadcasting
        if (Size > offset && x.dim() > 0 && y.dim() > 0) {
            // Create a tensor with a different shape for broadcasting
            std::vector<int64_t> broadcast_shape;
            if (x.dim() > 1) {
                broadcast_shape.push_back(x.size(0));
                broadcast_shape.push_back(1);  // This will force broadcasting
            } else {
                broadcast_shape.push_back(1);
            }
            
            torch::Tensor broadcast_tensor = torch::ones(broadcast_shape, x.options());
            
            // Test xlogy with broadcasting
            torch::Tensor result_broadcast = torch::special::xlogy(x, broadcast_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
