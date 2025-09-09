#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least enough data for tensor dimensions and some values
        if (Size < 16) {
            return 0;
        }

        // Extract dimensions for a 2D matrix
        int rows = extractInt(Data, Size, offset) % 100 + 1; // 1 to 100
        int cols = extractInt(Data, Size, offset) % 100 + 1; // 1 to 100
        
        // Limit matrix size to prevent excessive memory usage
        if (rows > 50 || cols > 50) {
            rows = std::min(rows, 50);
            cols = std::min(cols, 50);
        }

        // Extract data type
        int dtype_idx = extractInt(Data, Size, offset) % 6;
        torch::ScalarType dtype;
        switch (dtype_idx) {
            case 0: dtype = torch::kFloat32; break;
            case 1: dtype = torch::kFloat64; break;
            case 2: dtype = torch::kInt32; break;
            case 3: dtype = torch::kInt64; break;
            case 4: dtype = torch::kInt16; break;
            default: dtype = torch::kInt8; break;
        }

        // Create tensor with random values
        torch::Tensor input;
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            input = torch::randn({rows, cols}, torch::TensorOptions().dtype(dtype));
            
            // Add some edge case values
            if (offset + 4 < Size) {
                int special_case = extractInt(Data, Size, offset) % 10;
                switch (special_case) {
                    case 0: // Add infinity
                        if (rows > 0 && cols > 0) {
                            input[0][0] = std::numeric_limits<float>::infinity();
                        }
                        break;
                    case 1: // Add negative infinity
                        if (rows > 0 && cols > 0) {
                            input[0][0] = -std::numeric_limits<float>::infinity();
                        }
                        break;
                    case 2: // Add NaN
                        if (rows > 0 && cols > 0) {
                            input[0][0] = std::numeric_limits<float>::quiet_NaN();
                        }
                        break;
                    case 3: // Very large values
                        input = input * 1e10;
                        break;
                    case 4: // Very small values
                        input = input * 1e-10;
                        break;
                    case 5: // Zero matrix
                        input = torch::zeros({rows, cols}, torch::TensorOptions().dtype(dtype));
                        break;
                    case 6: // Identity-like matrix (ones on diagonal)
                        input = torch::zeros({rows, cols}, torch::TensorOptions().dtype(dtype));
                        for (int i = 0; i < std::min(rows, cols); ++i) {
                            input[i][i] = 1.0;
                        }
                        break;
                    default:
                        break;
                }
            }
        } else {
            // Integer types
            input = torch::randint(-1000, 1000, {rows, cols}, torch::TensorOptions().dtype(dtype));
            
            // Add some edge cases for integers
            if (offset + 4 < Size) {
                int special_case = extractInt(Data, Size, offset) % 6;
                switch (special_case) {
                    case 0: // Max values
                        input = torch::full({rows, cols}, 1000, torch::TensorOptions().dtype(dtype));
                        break;
                    case 1: // Min values
                        input = torch::full({rows, cols}, -1000, torch::TensorOptions().dtype(dtype));
                        break;
                    case 2: // Zero matrix
                        input = torch::zeros({rows, cols}, torch::TensorOptions().dtype(dtype));
                        break;
                    case 3: // Identity-like matrix
                        input = torch::zeros({rows, cols}, torch::TensorOptions().dtype(dtype));
                        for (int i = 0; i < std::min(rows, cols); ++i) {
                            input[i][i] = 1;
                        }
                        break;
                    default:
                        break;
                }
            }
        }

        // Test different tensor properties
        if (offset + 4 < Size) {
            int property_test = extractInt(Data, Size, offset) % 8;
            switch (property_test) {
                case 0: // Test with requires_grad
                    if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                        input = input.requires_grad_(true);
                    }
                    break;
                case 1: // Test with different memory layout
                    input = input.contiguous();
                    break;
                case 2: // Test with transposed tensor
                    input = input.t();
                    break;
                case 3: // Test with non-contiguous tensor
                    if (rows > 1 && cols > 1) {
                        input = input.slice(0, 0, rows, 2).slice(1, 0, cols, 2);
                    }
                    break;
                case 4: // Test with different device (if CUDA available)
                    if (torch::cuda::is_available()) {
                        input = input.to(torch::kCUDA);
                    }
                    break;
                default:
                    break;
            }
        }

        // Call torch::trace
        torch::Tensor result = torch::trace(input);

        // Verify result properties
        if (result.dim() != 0) {
            std::cerr << "Unexpected result dimension: " << result.dim() << std::endl;
        }

        // Test that result is finite for finite inputs (when applicable)
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            if (torch::all(torch::isfinite(input)).item<bool>()) {
                if (!torch::isfinite(result).item<bool>()) {
                    std::cerr << "Result should be finite for finite input" << std::endl;
                }
            }
        }

        // Additional edge cases
        if (offset + 4 < Size) {
            int edge_case = extractInt(Data, Size, offset) % 5;
            switch (edge_case) {
                case 0: // Test 1x1 matrix
                    {
                        torch::Tensor small_input = torch::randn({1, 1}, torch::TensorOptions().dtype(dtype));
                        torch::Tensor small_result = torch::trace(small_input);
                        // For 1x1 matrix, trace should equal the single element
                        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
                            if (torch::allclose(small_result, small_input[0][0]).item<bool>() == false) {
                                std::cerr << "Trace of 1x1 matrix should equal the element" << std::endl;
                            }
                        }
                    }
                    break;
                case 1: // Test rectangular matrix (more rows)
                    if (rows != cols) {
                        torch::Tensor rect_input = torch::randn({rows + 5, cols}, torch::TensorOptions().dtype(dtype));
                        torch::trace(rect_input);
                    }
                    break;
                case 2: // Test rectangular matrix (more cols)
                    if (rows != cols) {
                        torch::Tensor rect_input = torch::randn({rows, cols + 5}, torch::TensorOptions().dtype(dtype));
                        torch::trace(rect_input);
                    }
                    break;
                case 3: // Test with cloned tensor
                    {
                        torch::Tensor cloned_input = input.clone();
                        torch::trace(cloned_input);
                    }
                    break;
                case 4: // Test with detached tensor
                    if (input.requires_grad()) {
                        torch::Tensor detached_input = input.detach();
                        torch::trace(detached_input);
                    }
                    break;
                default:
                    break;
            }
        }

        // Force evaluation to catch any lazy evaluation issues
        result.item();

    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl; // do not change this, I need to know the exception.
        return -1; // discard the input
    }
    return 0; // keep the input
}