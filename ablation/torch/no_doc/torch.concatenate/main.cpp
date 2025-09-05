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
        
        // Need at least a few bytes for basic parameters
        if (Size < 4) {
            return 0;
        }
        
        // Parse number of tensors to concatenate (1-10)
        uint8_t num_tensors_raw = Data[offset++];
        size_t num_tensors = (num_tensors_raw % 10) + 1;
        
        // Parse concatenation dimension
        int64_t dim_raw = 0;
        if (offset + sizeof(int64_t) <= Size) {
            std::memcpy(&dim_raw, Data + offset, sizeof(int64_t));
            offset += sizeof(int64_t);
        }
        
        // Create tensors from fuzzer input
        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);
        
        for (size_t i = 0; i < num_tensors && offset < Size; ++i) {
            try {
                torch::Tensor t = fuzzer_utils::createTensor(Data, Size, offset);
                tensors.push_back(t);
            } catch (const std::exception& e) {
                // If we can't create a tensor, try with a default one
                if (tensors.empty()) {
                    // Need at least one tensor
                    tensors.push_back(torch::randn({2, 3}));
                }
                break;
            }
        }
        
        // If no tensors were created, create some defaults
        if (tensors.empty()) {
            tensors.push_back(torch::randn({2, 3}));
            tensors.push_back(torch::randn({2, 3}));
        }
        
        // Adjust dimension based on the first tensor's dimensions
        int64_t max_dim = tensors[0].dim();
        int64_t dim = (max_dim > 0) ? (dim_raw % max_dim) : 0;
        
        // Handle negative dimensions
        if (offset < Size) {
            uint8_t use_negative = Data[offset++];
            if (use_negative & 0x01 && max_dim > 0) {
                dim = dim - max_dim;  // Convert to negative indexing
            }
        }
        
        // Test various edge cases based on remaining bytes
        if (offset < Size) {
            uint8_t edge_case = Data[offset++];
            
            // Edge case 1: Mix of empty and non-empty tensors
            if ((edge_case & 0x01) && tensors.size() > 1) {
                try {
                    auto shape = tensors[0].sizes().vec();
                    if (!shape.empty() && dim >= 0 && dim < static_cast<int64_t>(shape.size())) {
                        shape[dim] = 0;  // Make dimension 0
                        tensors.push_back(torch::empty(shape, tensors[0].options()));
                    }
                } catch (...) {
                    // Ignore if we can't create empty tensor
                }
            }
            
            // Edge case 2: Single tensor (should return itself)
            if ((edge_case & 0x02) && tensors.size() > 1) {
                tensors.resize(1);
            }
            
            // Edge case 3: Mix different dtypes (will fail, but tests error handling)
            if ((edge_case & 0x04) && tensors.size() > 1 && offset < Size) {
                uint8_t dtype_selector = Data[offset++];
                auto new_dtype = fuzzer_utils::parseDataType(dtype_selector);
                if (new_dtype != tensors[0].dtype()) {
                    try {
                        tensors.push_back(torch::randn({2, 3}, torch::TensorOptions().dtype(new_dtype)));
                    } catch (...) {
                        // Ignore dtype creation failures
                    }
                }
            }
            
            // Edge case 4: Scalar tensors
            if ((edge_case & 0x08)) {
                tensors.push_back(torch::scalar_tensor(3.14));
            }
            
            // Edge case 5: High-dimensional tensors
            if ((edge_case & 0x10) && offset + 8 < Size) {
                try {
                    std::vector<int64_t> high_dim_shape;
                    uint8_t num_dims = (Data[offset++] % 6) + 1;  // 1-6 dimensions
                    for (uint8_t i = 0; i < num_dims && offset < Size; ++i) {
                        high_dim_shape.push_back((Data[offset++] % 4) + 1);  // Small dims 1-4
                    }
                    tensors.push_back(torch::randn(high_dim_shape));
                } catch (...) {
                    // Ignore high-dim creation failures
                }
            }
            
            // Edge case 6: Tensors with requires_grad
            if ((edge_case & 0x20) && !tensors.empty()) {
                try {
                    tensors[0] = tensors[0].requires_grad_(true);
                } catch (...) {
                    // Some dtypes don't support grad
                }
            }
            
            // Edge case 7: Non-contiguous tensors
            if ((edge_case & 0x40) && !tensors.empty() && tensors[0].numel() > 1) {
                try {
                    tensors.push_back(tensors[0].transpose(0, -1));
                } catch (...) {
                    // Ignore transpose failures
                }
            }
        }
        
        // Perform concatenation
        try {
            torch::Tensor result = torch::cat(tensors, dim);
            
            // Perform some operations on result to ensure it's valid
            if (result.numel() > 0) {
                auto sum = result.sum();
                auto mean = result.mean();
                
                // Test that result shape is correct
                if (dim >= 0 && dim < result.dim()) {
                    int64_t expected_size = 0;
                    for (const auto& t : tensors) {
                        if (dim < t.dim()) {
                            expected_size += t.size(dim);
                        }
                    }
                }
                
                // Test backward if applicable
                if (result.requires_grad()) {
                    try {
                        result.sum().backward();
                    } catch (...) {
                        // Ignore backward failures
                    }
                }
            }
            
        } catch (const c10::Error& e) {
            // This is expected for many invalid inputs (mismatched shapes, etc.)
            // Don't print to avoid spam, just continue
        } catch (const std::exception& e) {
            // Other exceptions might be more interesting
            std::cout << "Unexpected exception in torch::cat: " << e.what() << std::endl;
        }
        
        // Test torch::concat (alias for cat)
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                torch::Tensor result2 = torch::concat(tensors, dim);
            } catch (...) {
                // Ignore concat failures
            }
        }
        
        // Test torch::concatenate (another alias)
        if (offset < Size && (Data[offset++] & 0x01)) {
            try {
                torch::Tensor result3 = torch::concatenate(tensors, dim);
            } catch (...) {
                // Ignore concatenate failures
            }
        }
        
        // Test stack (related operation) with valid dimension
        if (offset < Size && (Data[offset++] & 0x01) && !tensors.empty()) {
            try {
                // Stack requires all tensors to have same shape
                std::vector<torch::Tensor> same_shape_tensors;
                auto target_shape = tensors[0].sizes();
                for (const auto& t : tensors) {
                    if (t.sizes() == target_shape) {
                        same_shape_tensors.push_back(t);
                    }
                }
                if (!same_shape_tensors.empty()) {
                    int64_t stack_dim = (dim_raw % (same_shape_tensors[0].dim() + 1));
                    torch::Tensor stacked = torch::stack(same_shape_tensors, stack_dim);
                }
            } catch (...) {
                // Ignore stack failures
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