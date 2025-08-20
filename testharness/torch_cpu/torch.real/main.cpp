#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Skip if there's not enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create a tensor from the input data
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.real operation
        torch::Tensor real_part = torch::real(input_tensor);
        
        // Try to access the tensor data to ensure it's valid
        if (real_part.defined()) {
            real_part.item();
        }
        
        // Try additional edge cases if we have more data
        if (offset + 1 < Size) {
            // Create a view of the tensor to test real on views
            torch::Tensor view_tensor;
            
            // Get a byte to determine what kind of view to create
            uint8_t view_type = Data[offset++];
            
            if (input_tensor.dim() > 0) {
                if (view_type % 3 == 0) {
                    // Create a slice view
                    view_tensor = input_tensor.slice(0, 0);
                } else if (view_type % 3 == 1 && input_tensor.numel() > 0) {
                    // Create a reshape view if possible
                    view_tensor = input_tensor.reshape({-1});
                } else {
                    // Create a transpose view
                    view_tensor = input_tensor.transpose(0, input_tensor.dim() - 1);
                }
                
                // Apply real to the view
                torch::Tensor real_view = torch::real(view_tensor);
                
                // Access the tensor data
                if (real_view.defined() && real_view.numel() > 0) {
                    real_view.item();
                }
            }
        }
        
        // Test with requires_grad if we have a floating point tensor
        if (input_tensor.is_floating_point() || input_tensor.is_complex()) {
            auto grad_tensor = input_tensor.detach().clone().requires_grad_(true);
            auto real_grad = torch::real(grad_tensor);
            
            // If tensor has elements, try backward pass
            if (real_grad.numel() > 0) {
                try {
                    real_grad.sum().backward();
                } catch (...) {
                    // Ignore backward errors
                }
            }
        }
        
        // Test with non-contiguous tensor if possible
        if (input_tensor.dim() >= 2) {
            auto non_contig = input_tensor.transpose(0, 1);
            if (!non_contig.is_contiguous()) {
                auto real_non_contig = torch::real(non_contig);
                
                if (real_non_contig.defined() && real_non_contig.numel() > 0) {
                    real_non_contig.item();
                }
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