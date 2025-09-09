#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic data for number of tensors and their properties
        if (Size < 16) return 0;

        // Extract number of tensors to stack (1-10 tensors)
        uint8_t num_tensors = (Data[offset++] % 10) + 1;
        
        std::vector<torch::Tensor> tensors;
        tensors.reserve(num_tensors);

        for (uint8_t i = 0; i < num_tensors; ++i) {
            if (offset >= Size) break;

            // Extract tensor properties
            uint8_t ndim = (Data[offset++] % 4) + 1; // 1-4 dimensions
            if (offset + ndim * 2 >= Size) break;

            std::vector<int64_t> shape;
            shape.reserve(ndim);
            
            for (uint8_t d = 0; d < ndim; ++d) {
                if (offset + 1 >= Size) break;
                // Small dimensions to avoid memory issues
                int64_t dim_size = (Data[offset++] % 8) + 1; // 1-8
                shape.push_back(dim_size);
            }

            if (shape.size() != ndim) break;

            // Extract dtype
            if (offset >= Size) break;
            torch::ScalarType dtype = static_cast<torch::ScalarType>((Data[offset++] % 11) + 1); // Skip Undefined
            
            // Create tensor with random data
            torch::Tensor tensor;
            try {
                tensor = torch::randn(shape, torch::TensorOptions().dtype(dtype));
                
                // Add some edge case values
                if (offset < Size) {
                    uint8_t special_val = Data[offset++] % 4;
                    switch (special_val) {
                        case 0: // Keep random values
                            break;
                        case 1: // Fill with zeros
                            tensor.zero_();
                            break;
                        case 2: // Fill with ones
                            tensor.fill_(1.0);
                            break;
                        case 3: // Fill with large values
                            tensor.fill_(1e6);
                            break;
                    }
                }
                
                tensors.push_back(tensor);
            } catch (...) {
                // Skip invalid tensor creation
                continue;
            }
        }

        // Need at least one tensor
        if (tensors.empty()) return 0;

        // Test torch::dstack with the tensor list
        torch::Tensor result = torch::dstack(tensors);

        // Verify result properties
        if (result.defined()) {
            auto result_shape = result.sizes();
            // dstack should create at least 3D tensor
            if (result_shape.size() >= 3) {
                // Access some elements to trigger potential issues
                if (result.numel() > 0) {
                    auto flat = result.flatten();
                    if (flat.numel() > 0) {
                        flat[0].item<double>(); // Access first element
                        if (flat.numel() > 1) {
                            flat[flat.numel() - 1].item<double>(); // Access last element
                        }
                    }
                }
            }
        }

        // Test with output tensor if we have enough data
        if (offset < Size && result.defined()) {
            try {
                torch::Tensor out = torch::empty_like(result);
                torch::dstack_out(out, tensors);
                
                // Verify output tensor
                if (out.numel() > 0) {
                    auto flat_out = out.flatten();
                    if (flat_out.numel() > 0) {
                        flat_out[0].item<double>();
                    }
                }
            } catch (...) {
                // Output tensor creation/usage might fail, continue
            }
        }

        // Test edge cases with different tensor combinations
        if (tensors.size() >= 2 && offset < Size) {
            // Test with subset of tensors
            uint8_t subset_size = (Data[offset++] % tensors.size()) + 1;
            std::vector<torch::Tensor> subset(tensors.begin(), tensors.begin() + subset_size);
            torch::Tensor subset_result = torch::dstack(subset);
            
            if (subset_result.defined() && subset_result.numel() > 0) {
                subset_result.sum().item<double>();
            }
        }

        // Test with single tensor
        if (!tensors.empty()) {
            std::vector<torch::Tensor> single_tensor = {tensors[0]};
            torch::Tensor single_result = torch::dstack(single_tensor);
            
            if (single_result.defined() && single_result.numel() > 0) {
                single_result.mean().item<double>();
            }
        }

        // Test with tensors of different shapes (should work due to atleast_3d behavior)
        if (tensors.size() >= 2 && offset < Size) {
            try {
                // Create tensors with different dimensionalities
                std::vector<torch::Tensor> mixed_tensors;
                
                // 1D tensor
                mixed_tensors.push_back(torch::randn({3}));
                
                // 2D tensor  
                mixed_tensors.push_back(torch::randn({3, 1}));
                
                // 3D tensor
                mixed_tensors.push_back(torch::randn({3, 1, 1}));
                
                torch::Tensor mixed_result = torch::dstack(mixed_tensors);
                if (mixed_result.defined() && mixed_result.numel() > 0) {
                    mixed_result.std().item<double>();
                }
            } catch (...) {
                // Mixed tensor stacking might fail in some cases
            }
        }

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}