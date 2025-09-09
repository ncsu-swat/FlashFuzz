#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least basic parameters
        if (Size < 20) return 0;

        // Extract tensor dimensions and properties
        auto ndims = extract_int(Data, Size, offset, 1, 4);
        std::vector<int64_t> dims;
        for (int i = 0; i < ndims; i++) {
            dims.push_back(extract_int(Data, Size, offset, 1, 10));
        }

        // Extract dtype
        auto dtype_idx = extract_int(Data, Size, offset, 0, 3);
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            default: dtype = torch::kFloat32; break;
        }

        // Create input tensor
        auto input = torch::randn(dims, torch::TensorOptions().dtype(dtype));
        
        // Add some variation to input values
        if (offset < Size) {
            auto scale = extract_float(Data, Size, offset, 0.1f, 10.0f);
            input = input * scale;
        }

        // Extract spacing parameter (optional)
        torch::Tensor spacing;
        bool use_spacing = extract_int(Data, Size, offset, 0, 1) == 1;
        if (use_spacing && input.numel() > 1) {
            auto spacing_size = std::min(static_cast<int64_t>(input.size(-1)), static_cast<int64_t>(5));
            spacing = torch::rand({spacing_size}, torch::TensorOptions().dtype(torch::kFloat32));
            // Ensure positive spacing
            spacing = torch::abs(spacing) + 0.01f;
        }

        // Extract dim parameter
        int64_t dim = -1; // default to last dimension
        if (input.dim() > 1) {
            dim = extract_int(Data, Size, offset, -input.dim(), input.dim() - 1);
        }

        // Extract edge_order parameter
        auto edge_order = extract_int(Data, Size, offset, 1, 2);

        // Test different gradient computation scenarios
        torch::Tensor result;

        // Scenario 1: Basic gradient without spacing
        if (!use_spacing) {
            result = torch::gradient(input, /*spacing=*/{}, dim, edge_order);
        } else {
            // Scenario 2: Gradient with spacing
            result = torch::gradient(input, spacing, dim, edge_order);
        }

        // Test edge cases
        if (offset < Size) {
            auto test_case = extract_int(Data, Size, offset, 0, 4);
            
            switch (test_case) {
                case 0: {
                    // Test with 1D tensor
                    auto tensor_1d = torch::randn({5}, torch::TensorOptions().dtype(dtype));
                    auto grad_1d = torch::gradient(tensor_1d);
                    break;
                }
                case 1: {
                    // Test with very small tensor
                    if (input.numel() >= 2) {
                        auto small_tensor = input.slice(0, 0, 2);
                        auto grad_small = torch::gradient(small_tensor, /*spacing=*/{}, -1, 1);
                    }
                    break;
                }
                case 2: {
                    // Test with different edge orders
                    if (input.numel() >= 3) {
                        auto grad_order1 = torch::gradient(input, /*spacing=*/{}, dim, 1);
                        auto grad_order2 = torch::gradient(input, /*spacing=*/{}, dim, 2);
                    }
                    break;
                }
                case 3: {
                    // Test with multiple dimensions
                    if (input.dim() >= 2) {
                        for (int64_t d = 0; d < input.dim(); d++) {
                            auto grad_dim = torch::gradient(input, /*spacing=*/{}, d, edge_order);
                        }
                    }
                    break;
                }
                case 4: {
                    // Test with uniform spacing
                    if (use_spacing && input.size(-1) > 1) {
                        auto uniform_spacing = torch::ones({input.size(dim)}, torch::TensorOptions().dtype(torch::kFloat32));
                        auto grad_uniform = torch::gradient(input, uniform_spacing, dim, edge_order);
                    }
                    break;
                }
            }
        }

        // Verify result properties
        if (result.defined()) {
            // Check that result has same shape as input
            if (result.sizes() != input.sizes()) {
                std::cerr << "Gradient result shape mismatch" << std::endl;
            }
            
            // Check for NaN or Inf values
            if (torch::any(torch::isnan(result)).item<bool>() || 
                torch::any(torch::isinf(result)).item<bool>()) {
                std::cerr << "Gradient contains NaN or Inf values" << std::endl;
            }
        }

        // Test error conditions
        if (offset < Size) {
            auto error_test = extract_int(Data, Size, offset, 0, 2);
            
            try {
                switch (error_test) {
                    case 0: {
                        // Test with invalid dimension
                        if (input.dim() > 0) {
                            auto invalid_dim = input.dim() + 5;
                            torch::gradient(input, /*spacing=*/{}, invalid_dim, edge_order);
                        }
                        break;
                    }
                    case 1: {
                        // Test with mismatched spacing size
                        if (input.dim() > 0 && input.size(-1) > 1) {
                            auto wrong_spacing = torch::ones({input.size(-1) + 2}, torch::TensorOptions().dtype(torch::kFloat32));
                            torch::gradient(input, wrong_spacing, -1, edge_order);
                        }
                        break;
                    }
                    case 2: {
                        // Test with invalid edge_order
                        torch::gradient(input, /*spacing=*/{}, dim, 0); // edge_order must be 1 or 2
                        break;
                    }
                }
            } catch (const std::exception&) {
                // Expected for invalid inputs
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