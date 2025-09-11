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
        
        // Create the source tensor
        torch::Tensor source_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the target tensor with different shape
        torch::Tensor target_tensor;
        if (offset < Size) {
            target_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we've consumed all data, create a tensor with different shape
            std::vector<int64_t> shape = {2, 3, 4};
            target_tensor = torch::empty(shape);
        }
        
        // Clone the source tensor to preserve original for comparison
        torch::Tensor source_clone = source_tensor.clone();
        
        // Apply resize_as_ operation
        source_tensor.resize_as_(target_tensor);
        
        // Verify the resize worked correctly
        if (!source_tensor.sizes().equals(target_tensor.sizes())) {
            throw std::runtime_error("resize_as_ failed: shapes don't match");
        }
        
        // Test edge cases if we have more data
        if (offset + 2 < Size) {
            // Create an empty tensor
            std::vector<int64_t> empty_shape = {0};
            torch::Tensor empty_tensor = torch::empty(empty_shape);
            
            // Test resizing to empty tensor
            torch::Tensor test1 = source_clone.clone();
            test1.resize_as_(empty_tensor);
            
            // Test resizing empty tensor to non-empty
            torch::Tensor test2 = empty_tensor.clone();
            test2.resize_as_(target_tensor);
        }
        
        // Test with scalar tensor if we have more data
        if (offset + 2 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(3.14);
            
            // Resize tensor to scalar
            torch::Tensor test3 = source_clone.clone();
            test3.resize_as_(scalar_tensor);
            
            // Resize scalar to tensor
            torch::Tensor test4 = scalar_tensor.clone();
            test4.resize_as_(target_tensor);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
