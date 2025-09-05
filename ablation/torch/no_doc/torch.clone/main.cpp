#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation and clone options
        if (Size < 3) {
            return 0;  // Not enough data, but keep for coverage
        }

        // Create the source tensor
        torch::Tensor tensor;
        try {
            tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create a tensor, try with a minimal default
            tensor = torch::randn({1});
            offset = Size; // Mark all data consumed
        }

        // If we have remaining bytes, use them to control clone behavior
        if (offset < Size) {
            uint8_t clone_options = Data[offset++];
            
            // Test different memory formats based on fuzzer input
            uint8_t memory_format_selector = clone_options & 0x07;  // 3 bits for format selection
            torch::MemoryFormat memory_format = torch::MemoryFormat::Contiguous;
            
            switch (memory_format_selector) {
                case 0:
                    memory_format = torch::MemoryFormat::Contiguous;
                    break;
                case 1:
                    memory_format = torch::MemoryFormat::Preserve;
                    break;
                case 2:
                    // ChannelsLast only makes sense for 4D tensors
                    if (tensor.dim() == 4) {
                        memory_format = torch::MemoryFormat::ChannelsLast;
                    }
                    break;
                case 3:
                    // ChannelsLast3d only makes sense for 5D tensors
                    if (tensor.dim() == 5) {
                        memory_format = torch::MemoryFormat::ChannelsLast3d;
                    }
                    break;
                default:
                    memory_format = torch::MemoryFormat::Preserve;
                    break;
            }

            // Clone with specified memory format
            torch::Tensor cloned;
            try {
                cloned = tensor.clone(memory_format);
                
                // Verify clone properties
                if (cloned.data_ptr() == tensor.data_ptr()) {
                    // This shouldn't happen - clone should create new storage
                    std::cerr << "Warning: Clone shares data pointer with original!" << std::endl;
                }
                
                // Test that modifications don't affect original
                if (cloned.numel() > 0 && cloned.dtype() != torch::kBool) {
                    cloned.mul_(2.0);  // Modify clone
                    
                    // Check original is unchanged
                    if (torch::allclose(tensor, cloned)) {
                        // They shouldn't be equal after modification unless tensor was all zeros
                        if (!torch::allclose(tensor, torch::zeros_like(tensor))) {
                            std::cerr << "Warning: Clone modification affected original!" << std::endl;
                        }
                    }
                }
            } catch (const c10::Error& e) {
                // Some memory format combinations might not be valid
                // Just try regular clone
                cloned = tensor.clone();
            }

            // Test additional clone variations if we have more data
            if (offset < Size) {
                uint8_t extra_tests = Data[offset++];
                
                // Test cloning non-contiguous tensors
                if (extra_tests & 0x01) {
                    if (tensor.dim() >= 2 && tensor.size(0) > 1 && tensor.size(1) > 1) {
                        try {
                            // Create a non-contiguous view via transpose
                            torch::Tensor transposed = tensor.transpose(0, 1);
                            torch::Tensor cloned_transposed = transposed.clone();
                            
                            // Cloned version should be contiguous
                            if (!cloned_transposed.is_contiguous()) {
                                std::cerr << "Warning: Clone of non-contiguous tensor is not contiguous!" << std::endl;
                            }
                        } catch (const c10::Error& e) {
                            // Transpose might fail for certain tensor configurations
                        }
                    }
                }
                
                // Test cloning sliced tensors
                if (extra_tests & 0x02) {
                    if (tensor.numel() > 2) {
                        try {
                            // Create a view via slicing
                            torch::Tensor sliced = tensor.flatten().slice(0, 0, tensor.numel() / 2);
                            torch::Tensor cloned_slice = sliced.clone();
                            
                            // Verify independence
                            if (cloned_slice.numel() > 0) {
                                sliced.fill_(1.0);
                                if (torch::allclose(cloned_slice, torch::ones_like(cloned_slice))) {
                                    std::cerr << "Warning: Clone of slice is not independent!" << std::endl;
                                }
                            }
                        } catch (const c10::Error& e) {
                            // Slicing operations might fail
                        }
                    }
                }
                
                // Test cloning with different devices if available
                if (extra_tests & 0x04) {
                    #ifdef USE_GPU
                    if (torch::cuda::is_available()) {
                        try {
                            torch::Tensor cuda_tensor = tensor.to(torch::kCUDA);
                            torch::Tensor cloned_cuda = cuda_tensor.clone();
                            
                            // Move back to CPU and verify
                            torch::Tensor cpu_clone = cloned_cuda.to(torch::kCPU);
                            if (!torch::allclose(tensor, cpu_clone, 1e-5, 1e-8)) {
                                std::cerr << "Warning: CUDA clone differs from original!" << std::endl;
                            }
                        } catch (const c10::Error& e) {
                            // CUDA operations might fail
                        }
                    }
                    #endif
                }
                
                // Test cloning with gradient tracking
                if (extra_tests & 0x08) {
                    if (tensor.dtype() == torch::kFloat || tensor.dtype() == torch::kDouble ||
                        tensor.dtype() == torch::kHalf || tensor.dtype() == torch::kBFloat16) {
                        try {
                            torch::Tensor grad_tensor = tensor.requires_grad_(true);
                            torch::Tensor cloned_grad = grad_tensor.clone();
                            
                            // Clone should preserve requires_grad
                            if (cloned_grad.requires_grad() != grad_tensor.requires_grad()) {
                                std::cerr << "Warning: Clone doesn't preserve requires_grad!" << std::endl;
                            }
                            
                            // But gradients should be independent
                            if (cloned_grad.numel() > 0) {
                                torch::Tensor sum1 = grad_tensor.sum();
                                sum1.backward();
                                
                                torch::Tensor sum2 = cloned_grad.sum();
                                sum2.backward();
                                
                                // Gradients should exist but be independent
                                if (grad_tensor.grad().defined() && cloned_grad.grad().defined()) {
                                    if (grad_tensor.grad().data_ptr() == cloned_grad.grad().data_ptr()) {
                                        std::cerr << "Warning: Cloned tensor shares gradient storage!" << std::endl;
                                    }
                                }
                            }
                        } catch (const c10::Error& e) {
                            // Gradient operations might fail
                        }
                    }
                }
                
                // Test cloning zero-strided tensors (broadcasting scenarios)
                if (extra_tests & 0x10) {
                    try {
                        if (tensor.numel() > 0) {
                            // Create a broadcasted view
                            torch::Tensor expanded = tensor.expand({-1});  // No-op expand
                            if (tensor.dim() >= 1 && tensor.size(0) == 1) {
                                expanded = tensor.expand({10});  // Actual broadcast
                            }
                            torch::Tensor cloned_expanded = expanded.clone();
                            
                            // Clone should materialize the broadcast
                            if (cloned_expanded.numel() != expanded.numel()) {
                                std::cerr << "Warning: Clone of expanded tensor has wrong size!" << std::endl;
                            }
                        }
                    } catch (const c10::Error& e) {
                        // Expand operations might fail
                    }
                }
            }
        } else {
            // No options specified, just do basic clone
            torch::Tensor cloned = tensor.clone();
            
            // Basic verification
            if (!torch::allclose(tensor, cloned, 1e-5, 1e-8)) {
                std::cerr << "Warning: Basic clone doesn't match original!" << std::endl;
                fuzzer_utils::compareTensors(tensor, cloned, Data, Size);
            }
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are expected for invalid operations
        return 0;  // Keep the input for coverage
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;  // Discard the input
    }
    catch (...)
    {
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;  // Discard the input
    }
    
    return 0;  // Keep the input
}