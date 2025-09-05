#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least 3 bytes: 2 for tensor creation + 1 for dim
        if (Size < 3) {
            return 0;
        }

        // Create primary tensor
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Parse dimension to unbind
        int64_t dim = 0;
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            
            // Handle both positive and negative dimensions
            if (tensor.dim() > 0) {
                // Map byte to valid dimension range [-rank, rank-1]
                int64_t rank = tensor.dim();
                int64_t total_range = 2 * rank;
                dim = (dim_byte % total_range) - rank;
                
                // Normalize negative dimensions
                if (dim < 0) {
                    dim = dim + rank;
                }
                // Ensure dim is in valid range [0, rank-1]
                dim = std::max(int64_t(0), std::min(dim, rank - 1));
            }
        }

#ifdef DEBUG_FUZZ
        std::cout << "Input tensor shape: " << tensor.sizes() 
                  << ", dtype: " << tensor.dtype() 
                  << ", dim to unbind: " << dim << std::endl;
#endif

        // Test unbind operation
        std::vector<torch::Tensor> unbinded;
        
        // Handle scalar tensor case (0D tensor)
        if (tensor.dim() == 0) {
            // unbind on scalar should fail, but let's try it
            try {
                unbinded = torch::unbind(tensor, dim);
            } catch (const c10::Error& e) {
                // Expected for scalar tensors
#ifdef DEBUG_FUZZ
                std::cout << "Expected error for scalar tensor unbind: " << e.what() << std::endl;
#endif
            }
        } else {
            // Normal unbind operation
            unbinded = torch::unbind(tensor, dim);
            
#ifdef DEBUG_FUZZ
            std::cout << "Unbind successful. Number of output tensors: " << unbinded.size() << std::endl;
            if (!unbinded.empty()) {
                std::cout << "First unbinded tensor shape: " << unbinded[0].sizes() << std::endl;
            }
#endif

            // Verify properties of unbinded tensors
            if (!unbinded.empty()) {
                // Check that number of unbinded tensors matches the dimension size
                int64_t expected_count = tensor.size(dim);
                if (unbinded.size() != static_cast<size_t>(expected_count)) {
                    std::cerr << "Unexpected number of unbinded tensors: " 
                              << unbinded.size() << " vs expected " << expected_count << std::endl;
                }
                
                // Check that each unbinded tensor has one less dimension
                for (size_t i = 0; i < unbinded.size(); ++i) {
                    if (unbinded[i].dim() != tensor.dim() - 1) {
                        std::cerr << "Unbinded tensor " << i << " has unexpected rank: "
                                  << unbinded[i].dim() << " vs expected " << (tensor.dim() - 1) << std::endl;
                    }
                    
                    // Verify data integrity by accessing elements
                    if (unbinded[i].numel() > 0) {
                        // Force computation by converting to CPU if needed
                        auto cpu_tensor = unbinded[i].cpu();
                        // Access first element to ensure data is valid
                        if (cpu_tensor.dtype() == torch::kFloat || cpu_tensor.dtype() == torch::kDouble) {
                            volatile auto val = cpu_tensor.flat<float>()[0];
                            (void)val; // Suppress unused warning
                        }
                    }
                }
            }
        }

        // Additional edge case testing with remaining bytes
        if (offset + 2 < Size) {
            // Try creating a non-contiguous tensor and unbinding it
            try {
                size_t offset2 = offset;
                torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset2);
                
                if (tensor2.dim() >= 2) {
                    // Make tensor non-contiguous by transposing
                    tensor2 = tensor2.transpose(0, tensor2.dim() - 1);
                    
                    // Parse another dimension for this tensor
                    int64_t dim2 = 0;
                    if (offset2 < Size) {
                        uint8_t dim_byte2 = Data[offset2++];
                        dim2 = dim_byte2 % tensor2.dim();
                    }
                    
                    auto unbinded2 = torch::unbind(tensor2, dim2);
                    
#ifdef DEBUG_FUZZ
                    std::cout << "Non-contiguous unbind successful. Count: " << unbinded2.size() << std::endl;
#endif
                }
            } catch (const std::exception& e) {
                // Secondary operations may fail, that's ok
#ifdef DEBUG_FUZZ
                std::cout << "Secondary unbind operation failed: " << e.what() << std::endl;
#endif
            }
        }

        // Test edge case: empty tensor unbind
        if (offset + 1 < Size) {
            uint8_t edge_selector = Data[offset++];
            if (edge_selector % 4 == 0) {
                // Create an empty tensor with specific shape
                std::vector<int64_t> empty_shape;
                uint8_t rank = (edge_selector / 4) % 4 + 1;
                for (uint8_t i = 0; i < rank; ++i) {
                    if (i == 0) {
                        empty_shape.push_back(0); // Make first dimension 0
                    } else {
                        empty_shape.push_back((edge_selector + i) % 5 + 1);
                    }
                }
                
                torch::Tensor empty_tensor = torch::zeros(empty_shape);
                try {
                    auto empty_unbinded = torch::unbind(empty_tensor, 0);
#ifdef DEBUG_FUZZ
                    std::cout << "Empty tensor unbind count: " << empty_unbinded.size() << std::endl;
#endif
                } catch (const std::exception& e) {
#ifdef DEBUG_FUZZ
                    std::cout << "Empty tensor unbind failed: " << e.what() << std::endl;
#endif
                }
            }
        }

        // Test with different memory layouts if we have more data
        if (offset + 3 < Size && tensor.dim() >= 2) {
            // Create strided view
            try {
                auto strided = tensor.as_strided(
                    {tensor.size(0) / 2, tensor.size(1)},
                    {tensor.stride(0) * 2, tensor.stride(1)}
                );
                
                auto strided_unbinded = torch::unbind(strided, 0);
#ifdef DEBUG_FUZZ
                std::cout << "Strided tensor unbind count: " << strided_unbinded.size() << std::endl;
#endif
            } catch (const std::exception& e) {
                // Strided operations may fail for various valid reasons
#ifdef DEBUG_FUZZ
                std::cout << "Strided unbind failed: " << e.what() << std::endl;
#endif
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}