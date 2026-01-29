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
        
        // Create the source tensor
        torch::Tensor source_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create the target tensor with potentially different shape
        torch::Tensor target_tensor;
        if (offset < Size) {
            target_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } else {
            // If we've consumed all data, create a tensor with different shape
            std::vector<int64_t> shape = {2, 3, 4};
            target_tensor = torch::empty(shape);
        }
        
        // Clone the source tensor for additional tests
        torch::Tensor source_clone = source_tensor.clone();
        
        // Test 1: Basic resize_as_ operation
        source_tensor.resize_as_(target_tensor);
        
        // Test 2: With memory_format option based on fuzzer data
        if (offset < Size) {
            uint8_t format_selector = Data[offset % Size];
            offset++;
            
            torch::Tensor test_tensor = source_clone.clone();
            
            try {
                // Test different memory formats
                if (format_selector % 3 == 0) {
                    test_tensor.resize_as_(target_tensor, c10::MemoryFormat::Contiguous);
                } else if (format_selector % 3 == 1) {
                    // ChannelsLast only valid for 4D tensors
                    if (target_tensor.dim() == 4) {
                        test_tensor.resize_as_(target_tensor, c10::MemoryFormat::ChannelsLast);
                    } else {
                        test_tensor.resize_as_(target_tensor, c10::MemoryFormat::Contiguous);
                    }
                } else {
                    test_tensor.resize_as_(target_tensor, c10::MemoryFormat::Preserve);
                }
            } catch (const c10::Error&) {
                // Some memory format combinations may not be valid, that's expected
            }
        }
        
        // Test 3: Resize to empty tensor
        if (offset < Size) {
            try {
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape);
                
                torch::Tensor test1 = source_clone.clone();
                test1.resize_as_(empty_tensor);
            } catch (const c10::Error&) {
                // May fail for certain tensor types
            }
        }
        
        // Test 4: Resize empty tensor to non-empty
        if (offset < Size) {
            try {
                std::vector<int64_t> empty_shape = {0};
                torch::Tensor empty_tensor = torch::empty(empty_shape);
                empty_tensor.resize_as_(target_tensor);
            } catch (const c10::Error&) {
                // May fail for certain configurations
            }
        }
        
        // Test 5: Resize to/from scalar tensor
        if (offset < Size) {
            try {
                torch::Tensor scalar_tensor = torch::tensor(3.14);
                
                // Resize tensor to scalar shape
                torch::Tensor test3 = source_clone.clone();
                test3.resize_as_(scalar_tensor);
            } catch (const c10::Error&) {
                // Expected for some tensor configurations
            }
            
            try {
                torch::Tensor scalar_tensor = torch::tensor(3.14);
                // Resize scalar to tensor shape
                scalar_tensor.resize_as_(target_tensor);
            } catch (const c10::Error&) {
                // Expected for some tensor configurations
            }
        }
        
        // Test 6: Resize with same shape (no-op case)
        if (offset < Size) {
            torch::Tensor same_shape = source_clone.clone();
            same_shape.resize_as_(source_clone);
        }
        
        // Test 7: Different dtypes
        if (offset < Size) {
            try {
                torch::Tensor float_tensor = torch::randn({2, 3});
                torch::Tensor int_tensor = torch::randint(0, 10, {4, 5});
                
                // resize_as_ doesn't change dtype, just shape
                float_tensor.resize_as_(int_tensor);
            } catch (const c10::Error&) {
                // May have restrictions
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;  // Tell libFuzzer to discard invalid input
    }
    return 0; // keep the input
}