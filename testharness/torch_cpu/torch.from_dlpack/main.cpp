#include "fuzzer_utils.h" // General fuzzing utilities
#include <ATen/DLConvertor.h>
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// Target API keyword: torch.from_dlpack

// Convert a tensor to DLPack and back using the C++ API.
static torch::Tensor roundtrip_dlpack(const torch::Tensor &tensor)
{
    DLManagedTensor *dlpack = at::toDLPack(tensor);
    torch::Tensor output = at::fromDLPack(dlpack);
    return output;
}

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
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Convert tensor to DLPack format and back
        torch::Tensor output_tensor = roundtrip_dlpack(input_tensor);
        
        // Verify the conversion worked correctly
        if (input_tensor.sizes() != output_tensor.sizes() ||
            input_tensor.dtype() != output_tensor.dtype()) {
            throw std::runtime_error("DLPack conversion changed tensor properties");
        }
        
        // Test with different tensor properties
        if (offset + 1 < Size) {
            // Create another tensor with different properties
            torch::Tensor another_tensor = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Try to convert it to DLPack and back
            torch::Tensor another_output = roundtrip_dlpack(another_tensor);
            
            // Try operations on the converted tensor
            if (another_output.numel() > 0) {
                try {
                    auto sum = another_output.sum();
                    auto mean = another_output.mean();
                    (void)sum;
                    (void)mean;
                } catch (...) {
                    // Some dtypes don't support mean/sum, ignore
                }
            }
        }
        
        // Test with empty tensor
        if (offset + 1 < Size) {
            try {
                torch::Tensor empty_tensor = torch::empty({0});
                torch::Tensor empty_output = roundtrip_dlpack(empty_tensor);
                (void)empty_output;
            } catch (...) {
                // Empty tensor DLPack conversion may fail, ignore
            }
        }
        
        // Test with scalar tensor
        if (offset + 1 < Size) {
            torch::Tensor scalar_tensor = torch::tensor(3.14);
            torch::Tensor scalar_output = roundtrip_dlpack(scalar_tensor);
            (void)scalar_output;
        }
        
        // Test with non-contiguous tensor (sliced)
        if (offset + 1 < Size) {
            try {
                torch::Tensor base_tensor = torch::randn({4, 4});
                torch::Tensor sliced_tensor = base_tensor.slice(0, 0, 2).slice(1, 0, 2);
                // DLPack requires contiguous tensors, so make it contiguous first
                torch::Tensor contiguous_sliced = sliced_tensor.contiguous();
                torch::Tensor sliced_output = roundtrip_dlpack(contiguous_sliced);
                (void)sliced_output;
            } catch (...) {
                // Ignore failures for edge cases
            }
        }
        
        // Test with different dtypes
        if (offset + 1 < Size) {
            uint8_t dtype_selector = Data[offset % Size];
            offset++;
            
            try {
                torch::Tensor typed_tensor;
                switch (dtype_selector % 6) {
                    case 0:
                        typed_tensor = torch::ones({2, 3}, torch::kFloat32);
                        break;
                    case 1:
                        typed_tensor = torch::ones({2, 3}, torch::kFloat64);
                        break;
                    case 2:
                        typed_tensor = torch::ones({2, 3}, torch::kInt32);
                        break;
                    case 3:
                        typed_tensor = torch::ones({2, 3}, torch::kInt64);
                        break;
                    case 4:
                        typed_tensor = torch::ones({2, 3}, torch::kInt16);
                        break;
                    case 5:
                        typed_tensor = torch::ones({2, 3}, torch::kUInt8);
                        break;
                }
                torch::Tensor typed_output = roundtrip_dlpack(typed_tensor);
                (void)typed_output;
            } catch (...) {
                // Some dtypes may not be supported by DLPack
            }
        }
        
        // Test with multi-dimensional tensor
        if (offset + 1 < Size) {
            try {
                torch::Tensor multi_dim = torch::randn({2, 3, 4, 5});
                torch::Tensor multi_dim_output = roundtrip_dlpack(multi_dim);
                (void)multi_dim_output;
            } catch (...) {
                // Ignore failures
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}