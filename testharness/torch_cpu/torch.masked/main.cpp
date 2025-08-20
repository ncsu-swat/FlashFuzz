#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create mask tensor (boolean type)
        torch::Tensor mask;
        if (offset < Size) {
            mask = fuzzer_utils::createTensor(Data, Size, offset);
            // Convert to boolean mask if not already
            if (mask.dtype() != torch::kBool) {
                mask = mask.to(torch::kBool);
            }
        } else {
            // Create a default mask if we don't have enough data
            mask = torch::ones_like(input_tensor, torch::kBool);
        }
        
        // Create value tensor for masked_fill
        torch::Tensor value;
        if (offset < Size) {
            value = fuzzer_utils::createTensor(Data, Size, offset);
            // Ensure value has compatible dtype with input_tensor
            if (value.dtype() != input_tensor.dtype()) {
                value = value.to(input_tensor.dtype());
            }
        } else {
            // Create a scalar tensor with a simple value
            value = torch::scalar_tensor(0, input_tensor.dtype());
        }
        
        // Test different masked operations
        
        // 1. masked_fill
        try {
            torch::Tensor result_fill = input_tensor.masked_fill(mask, value.item());
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 2. masked_fill with scalar value
        try {
            torch::Tensor result_fill_scalar = input_tensor.masked_fill(mask, 0);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 3. masked_select
        try {
            torch::Tensor result_select = torch::masked_select(input_tensor, mask);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 4. masked_scatter
        try {
            // Create a source tensor for scatter
            torch::Tensor source;
            if (offset < Size) {
                source = fuzzer_utils::createTensor(Data, Size, offset);
                if (source.dtype() != input_tensor.dtype()) {
                    source = source.to(input_tensor.dtype());
                }
            } else {
                source = torch::ones({1}, input_tensor.dtype());
            }
            
            torch::Tensor result_scatter = input_tensor.masked_scatter(mask, source);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 5. masked_fill_ (in-place version)
        try {
            torch::Tensor clone = input_tensor.clone();
            clone.masked_fill_(mask, value.item());
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 6. masked_scatter_ (in-place version)
        try {
            torch::Tensor clone = input_tensor.clone();
            torch::Tensor source;
            if (offset < Size) {
                source = fuzzer_utils::createTensor(Data, Size, offset);
                if (source.dtype() != clone.dtype()) {
                    source = source.to(clone.dtype());
                }
            } else {
                source = torch::ones({1}, clone.dtype());
            }
            
            clone.masked_scatter_(mask, source);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 7. Test with different mask shapes
        try {
            // Create a broadcast-compatible mask
            std::vector<int64_t> mask_shape;
            if (input_tensor.dim() > 0) {
                for (int i = 0; i < input_tensor.dim(); i++) {
                    if (i % 2 == 0 && input_tensor.size(i) > 1) {
                        mask_shape.push_back(1);  // Create broadcast dimension
                    } else {
                        mask_shape.push_back(input_tensor.size(i));
                    }
                }
                torch::Tensor broadcast_mask = torch::ones(mask_shape, torch::kBool);
                torch::Tensor result = input_tensor.masked_fill(broadcast_mask, 0);
            }
        } catch (const std::exception& e) {
            // Continue with other tests
        }
        
        // 8. Test with empty mask
        try {
            torch::Tensor empty_mask = torch::zeros({0}, torch::kBool);
            torch::Tensor result = torch::masked_select(input_tensor, empty_mask);
        } catch (const std::exception& e) {
            // Continue with other tests
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}