#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

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
        
        // Create a tensor to be nested
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Create a second tensor if we have more data
        torch::Tensor tensor2;
        if (offset + 4 < Size) {
            tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        }
        
        // Test different nested tensor operations
        
        // 1. Create a nested tensor from a list of tensors
        std::vector<torch::Tensor> tensor_list;
        tensor_list.push_back(tensor);
        
        if (tensor2.defined()) {
            tensor_list.push_back(tensor2);
        } else {
            // If we don't have a second tensor, duplicate the first one
            tensor_list.push_back(tensor);
        }
        
        // Create a nested tensor
        torch::Tensor nested_tensor = torch::nested::nested_tensor(tensor_list);
        
        // 2. Test nested tensor properties and operations
        auto is_nested = nested_tensor.is_nested();
        auto nested_size = nested_tensor.nested_size();
        auto nested_dim = nested_tensor.nested_dim();
        
        // 3. Test buffer conversion
        auto buffer = torch::nested::to_padded_tensor(nested_tensor, 0.0);
        
        // 4. Test unbind operation if we have enough dimensions
        if (nested_tensor.dim() > 0) {
            int64_t dim = 0;
            if (offset < Size) {
                // Use some data to determine the dimension
                dim = static_cast<int64_t>(Data[offset++]) % std::max(static_cast<int64_t>(1), nested_tensor.dim());
            }
            auto unbind_result = torch::unbind(nested_tensor, dim);
        }
        
        // 5. Test nested tensor creation with different options
        if (offset + 1 < Size) {
            uint8_t option_byte = Data[offset++];
            
            // Test with different padding values
            double padding_value = 0.0;
            if (option_byte & 0x01) {
                padding_value = -1.0;
            } else if (option_byte & 0x02) {
                padding_value = 1.0;
            }
            
            // Test with different layout options
            auto layout = torch::kStrided;
            if (option_byte & 0x04) {
                layout = torch::kSparse;
            }
            
            // Create a nested tensor with specific options
            auto options = torch::TensorOptions()
                .dtype(tensor.dtype())
                .layout(layout);
            
            torch::Tensor nested_tensor_with_options = torch::nested::nested_tensor(
                tensor_list, options);
            
            // Convert to padded tensor with the selected padding value
            auto padded = torch::nested::to_padded_tensor(nested_tensor_with_options, padding_value);
        }
        
        // 6. Test nested_tensor_from_mask if we have enough data
        if (offset + 4 < Size) {
            // Create a mask tensor
            torch::Tensor mask = fuzzer_utils::createTensor(Data, Size, offset);
            
            // Ensure mask is boolean
            mask = mask.to(torch::kBool);
            
            // Try to create a nested tensor from mask
            try {
                auto nested_from_mask = torch::_nested_tensor_from_mask(tensor, mask);
            } catch (const std::exception&) {
                // Ignore exceptions from invalid mask shapes
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