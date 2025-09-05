#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least minimal bytes for tensor creation and dim selection
        if (Size < 3) {
            return 0;
        }

        // Create input tensor from fuzzer data
        torch::Tensor input_tensor;
        try {
            input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        } catch (const std::exception& e) {
            // If we can't create a valid tensor, try with a simple default
            if (offset < Size) {
                uint8_t rank = Data[offset] % 5;  // 0-4 dimensions
                std::vector<int64_t> shape;
                for (int i = 0; i < rank; ++i) {
                    shape.push_back(1 + (i % 3));  // Small default shape
                }
                input_tensor = torch::randn(shape);
            } else {
                return 0;
            }
        }

        // Get dimension to unbind along
        int64_t dim = 0;
        if (offset < Size) {
            uint8_t dim_byte = Data[offset++];
            
            // Handle various dimension selection strategies
            if (input_tensor.dim() > 0) {
                // Strategy 1: Use modulo to select valid dimension
                if (dim_byte < 128) {
                    dim = dim_byte % input_tensor.dim();
                } 
                // Strategy 2: Try negative indexing
                else {
                    dim = -(1 + (dim_byte % input_tensor.dim()));
                }
            }
        }

        // Special cases to increase coverage
        if (offset < Size && Data[offset++] % 4 == 0) {
            // Try with scalar tensor (dim=0)
            torch::Tensor scalar = torch::tensor(3.14);
            try {
                auto result = torch::unbind(scalar, 0);
            } catch (...) {
                // Expected to fail for scalar
            }
        }

        // Test with various tensor properties
        if (offset < Size) {
            uint8_t property_selector = Data[offset++];
            
            // Make tensor non-contiguous
            if (property_selector & 0x01) {
                if (input_tensor.dim() >= 2 && input_tensor.size(0) > 1 && input_tensor.size(1) > 1) {
                    input_tensor = input_tensor.transpose(0, 1);
                }
            }
            
            // Try with sliced tensor
            if (property_selector & 0x02) {
                if (input_tensor.dim() > 0 && input_tensor.size(0) > 2) {
                    input_tensor = input_tensor.slice(0, 0, 2);
                }
            }
            
            // Try with view
            if (property_selector & 0x04) {
                if (input_tensor.numel() > 0) {
                    try {
                        input_tensor = input_tensor.view({-1});
                        dim = 0;  // Reset dim for 1D tensor
                    } catch (...) {
                        // View might fail, continue with original
                    }
                }
            }
            
            // Try with requires_grad
            if (property_selector & 0x08) {
                if (input_tensor.dtype() == torch::kFloat || 
                    input_tensor.dtype() == torch::kDouble ||
                    input_tensor.dtype() == torch::kHalf ||
                    input_tensor.dtype() == torch::kBFloat16) {
                    input_tensor = input_tensor.requires_grad_(true);
                }
            }
        }

        // Main unbind operation
        std::vector<torch::Tensor> unbind_result;
        try {
            unbind_result = torch::unbind(input_tensor, dim);
            
            // Verify results
            if (!unbind_result.empty()) {
                // Check that unbinding worked correctly
                int64_t expected_num = (input_tensor.dim() > 0 && dim >= -input_tensor.dim() && dim < input_tensor.dim()) 
                                      ? input_tensor.size(dim >= 0 ? dim : dim + input_tensor.dim()) 
                                      : 0;
                
                if (unbind_result.size() != expected_num && expected_num > 0) {
                    std::cerr << "Unexpected number of unbind results" << std::endl;
                }
                
                // Access some results to ensure they're valid
                for (size_t i = 0; i < std::min(size_t(3), unbind_result.size()); ++i) {
                    auto& t = unbind_result[i];
                    // Trigger potential issues by accessing tensor properties
                    auto shape = t.sizes();
                    auto dtype = t.dtype();
                    auto device = t.device();
                    
                    // Try basic operations on unbind results
                    if (t.numel() > 0) {
                        if (t.dtype() == torch::kFloat || t.dtype() == torch::kDouble) {
                            auto sum = t.sum();
                        }
                    }
                }
            }
        } catch (const c10::Error& e) {
            // PyTorch-specific errors are expected for invalid operations
            // Continue fuzzing
        }

        // Edge cases with empty tensors
        if (offset < Size && Data[offset++] % 3 == 0) {
            // Empty tensor with various shapes
            std::vector<std::vector<int64_t>> empty_shapes = {
                {0},
                {0, 5},
                {5, 0},
                {0, 0},
                {2, 0, 3},
                {1, 1, 0, 1}
            };
            
            for (const auto& shape : empty_shapes) {
                try {
                    torch::Tensor empty_t = torch::empty(shape);
                    for (int d = -static_cast<int>(shape.size()); d < static_cast<int>(shape.size()); ++d) {
                        try {
                            auto result = torch::unbind(empty_t, d);
                        } catch (...) {
                            // Some dimensions might be invalid
                        }
                    }
                } catch (...) {
                    // Shape creation might fail
                }
            }
        }

        // Test with different memory layouts
        if (offset < Size && Data[offset++] % 2 == 0) {
            if (input_tensor.dim() >= 2) {
                // Create strided tensor
                try {
                    auto strided = input_tensor.as_strided(
                        input_tensor.sizes(),
                        input_tensor.strides()
                    );
                    auto result = torch::unbind(strided, 0);
                } catch (...) {
                    // Strided operation might fail
                }
            }
        }

        // Test unbind with all valid dimensions
        if (offset < Size && Data[offset++] % 5 == 0) {
            for (int64_t d = -input_tensor.dim(); d < input_tensor.dim(); ++d) {
                try {
                    auto result = torch::unbind(input_tensor, d);
                } catch (...) {
                    // Some dimensions might fail
                }
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