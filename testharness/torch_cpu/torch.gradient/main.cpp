#include "fuzzer_utils.h"
#include <iostream>
#include <vector>

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
        
        // Need at least some data to proceed
        if (Size < 4) {
            return 0;
        }
        
        // Create input tensor - needs to be floating point for gradient
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Ensure input is floating point (gradient requires it)
        if (!input.is_floating_point()) {
            input = input.to(torch::kFloat32);
        }
        
        // Ensure input has at least 2 elements in each dimension for gradient
        if (input.numel() < 2) {
            return 0;
        }
        
        // Parse edge_order parameter (1 or 2 are valid)
        int64_t edge_order = 1;
        if (offset < Size) {
            edge_order = (Data[offset++] % 2) + 1; // 1 or 2
        }
        
        // Parse dim parameter
        c10::optional<int64_t> dim = c10::nullopt;
        if (offset < Size && input.dim() > 0) {
            uint8_t use_dim = Data[offset++] % 2;
            if (use_dim && offset < Size) {
                int64_t d = static_cast<int64_t>(Data[offset++] % input.dim());
                // Only use this dim if tensor has >= 2 elements in this dimension
                if (input.size(d) >= 2) {
                    dim = d;
                }
            }
        }
        
        // Parse spacing parameter
        c10::optional<torch::Scalar> spacing = c10::nullopt;
        if (offset < Size) {
            uint8_t use_spacing = Data[offset++] % 2;
            if (use_spacing && offset < Size) {
                double scalar_spacing = static_cast<double>(Data[offset++] + 1) / 10.0; // Avoid zero
                spacing = torch::Scalar(scalar_spacing);
            }
        }
        
        std::vector<torch::Tensor> result;
        
        // Case 1: Basic gradient call (computes gradient along all dimensions)
        try {
            // Check all dimensions have size >= 2
            bool all_dims_valid = true;
            for (int64_t i = 0; i < input.dim(); ++i) {
                if (input.size(i) < 2) {
                    all_dims_valid = false;
                    break;
                }
            }
            if (all_dims_valid) {
                result = torch::gradient(input);
            }
        } catch (const std::exception&) {
            // Expected for some inputs
        }
        
        // Case 2: With spacing scalar only
        if (spacing.has_value()) {
            try {
                bool all_dims_valid = true;
                for (int64_t i = 0; i < input.dim(); ++i) {
                    if (input.size(i) < 2) {
                        all_dims_valid = false;
                        break;
                    }
                }
                if (all_dims_valid) {
                    result = torch::gradient(input, spacing);
                }
            } catch (const std::exception&) {
                // Expected for some inputs
            }
        }
        
        // Case 3: With dim only
        if (dim.has_value()) {
            try {
                result = torch::gradient(input, c10::nullopt, dim, edge_order);
            } catch (const std::exception&) {
                // Expected for some inputs
            }
        }
        
        // Case 4: With spacing and dim
        if (spacing.has_value() && dim.has_value()) {
            try {
                result = torch::gradient(input, spacing, dim, edge_order);
            } catch (const std::exception&) {
                // Expected for some inputs
            }
        }
        
        // Case 5: With edge_order=2 if different from parsed value
        if (edge_order == 1 && dim.has_value()) {
            try {
                result = torch::gradient(input, spacing, dim, 2);
            } catch (const std::exception&) {
                // Expected for some inputs
            }
        }
        
        // Case 6: Test with ArrayRef<Scalar> spacing
        if (offset + 1 < Size && input.dim() > 0) {
            try {
                std::vector<torch::Scalar> spacing_vec;
                int64_t target_dim = dim.has_value() ? dim.value() : 0;
                if (input.size(target_dim) >= 2) {
                    double sp = static_cast<double>(Data[offset++] + 1) / 10.0;
                    spacing_vec.push_back(torch::Scalar(sp));
                    
                    std::vector<int64_t> dim_vec = {target_dim};
                    result = torch::gradient(input, c10::ArrayRef<torch::Scalar>(spacing_vec), 
                                            c10::ArrayRef<int64_t>(dim_vec), edge_order);
                }
            } catch (const std::exception&) {
                // Expected for some inputs
            }
        }
        
        // Case 7: Test with Tensor spacing
        if (offset + 2 < Size && dim.has_value()) {
            try {
                int64_t target_dim = dim.value();
                int64_t dim_size = input.size(target_dim);
                if (dim_size >= 2) {
                    // Create coordinate tensor for this dimension
                    torch::Tensor coord = torch::arange(dim_size, input.options());
                    std::vector<torch::Tensor> spacing_tensors = {coord};
                    std::vector<int64_t> dim_vec = {target_dim};
                    
                    result = torch::gradient(input, c10::ArrayRef<torch::Tensor>(spacing_tensors),
                                            c10::ArrayRef<int64_t>(dim_vec), edge_order);
                }
            } catch (const std::exception&) {
                // Expected for some inputs
            }
        }
        
        // Verify results are valid if we got any
        for (const auto& r : result) {
            (void)r.numel();
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}