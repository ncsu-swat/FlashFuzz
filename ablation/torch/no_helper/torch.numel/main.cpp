#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;

        // Need at least some data to create a tensor
        if (Size < 4) {
            return 0;
        }

        // Extract number of dimensions (1-8 dimensions)
        int num_dims = (Data[offset] % 8) + 1;
        offset++;

        // Extract dimensions for the tensor
        std::vector<int64_t> dims;
        for (int i = 0; i < num_dims && offset < Size; i++) {
            // Limit dimension size to avoid memory issues
            int64_t dim_size = (Data[offset] % 10) + 1; // 1-10 per dimension
            dims.push_back(dim_size);
            offset++;
        }

        // If we don't have enough data, use default dimensions
        if (dims.empty()) {
            dims = {2, 3};
        }

        // Extract data type
        torch::ScalarType dtype = torch::kFloat32;
        if (offset < Size) {
            int dtype_choice = Data[offset] % 8;
            offset++;
            switch (dtype_choice) {
                case 0: dtype = torch::kFloat32; break;
                case 1: dtype = torch::kFloat64; break;
                case 2: dtype = torch::kInt32; break;
                case 3: dtype = torch::kInt64; break;
                case 4: dtype = torch::kInt8; break;
                case 5: dtype = torch::kUInt8; break;
                case 6: dtype = torch::kBool; break;
                case 7: dtype = torch::kFloat16; break;
                default: dtype = torch::kFloat32; break;
            }
        }

        // Create tensor with the specified dimensions and type
        torch::Tensor input_tensor;
        
        // Test different tensor creation methods
        if (offset < Size) {
            int creation_method = Data[offset] % 6;
            offset++;
            
            switch (creation_method) {
                case 0:
                    input_tensor = torch::zeros(dims, torch::TensorOptions().dtype(dtype));
                    break;
                case 1:
                    input_tensor = torch::ones(dims, torch::TensorOptions().dtype(dtype));
                    break;
                case 2:
                    input_tensor = torch::randn(dims, torch::TensorOptions().dtype(dtype));
                    break;
                case 3:
                    input_tensor = torch::rand(dims, torch::TensorOptions().dtype(dtype));
                    break;
                case 4:
                    input_tensor = torch::empty(dims, torch::TensorOptions().dtype(dtype));
                    break;
                case 5:
                    input_tensor = torch::full(dims, 42.0, torch::TensorOptions().dtype(dtype));
                    break;
                default:
                    input_tensor = torch::zeros(dims, torch::TensorOptions().dtype(dtype));
                    break;
            }
        } else {
            input_tensor = torch::zeros(dims, torch::TensorOptions().dtype(dtype));
        }

        // Test torch::numel with the created tensor
        int64_t num_elements = torch::numel(input_tensor);

        // Verify the result makes sense
        int64_t expected_numel = 1;
        for (auto dim : dims) {
            expected_numel *= dim;
        }

        // Basic sanity check
        if (num_elements != expected_numel) {
            std::cerr << "Unexpected numel result: got " << num_elements 
                      << ", expected " << expected_numel << std::endl;
        }

        // Test edge cases if we have more data
        if (offset < Size) {
            int edge_case = Data[offset] % 4;
            offset++;

            switch (edge_case) {
                case 0: {
                    // Test with empty tensor
                    torch::Tensor empty_tensor = torch::empty({0}, torch::TensorOptions().dtype(dtype));
                    int64_t empty_numel = torch::numel(empty_tensor);
                    if (empty_numel != 0) {
                        std::cerr << "Empty tensor numel should be 0, got " << empty_numel << std::endl;
                    }
                    break;
                }
                case 1: {
                    // Test with scalar tensor
                    torch::Tensor scalar_tensor = torch::tensor(42.0, torch::TensorOptions().dtype(dtype));
                    int64_t scalar_numel = torch::numel(scalar_tensor);
                    if (scalar_numel != 1) {
                        std::cerr << "Scalar tensor numel should be 1, got " << scalar_numel << std::endl;
                    }
                    break;
                }
                case 2: {
                    // Test with reshaped tensor
                    torch::Tensor reshaped = input_tensor.reshape({-1});
                    int64_t reshaped_numel = torch::numel(reshaped);
                    if (reshaped_numel != num_elements) {
                        std::cerr << "Reshaped tensor numel mismatch" << std::endl;
                    }
                    break;
                }
                case 3: {
                    // Test with sliced tensor
                    if (input_tensor.numel() > 1) {
                        torch::Tensor sliced = input_tensor.slice(0, 0, 1);
                        int64_t sliced_numel = torch::numel(sliced);
                        // Just ensure it doesn't crash
                    }
                    break;
                }
            }
        }

        // Test with different tensor properties if more data available
        if (offset < Size) {
            int property_test = Data[offset] % 3;
            offset++;

            switch (property_test) {
                case 0: {
                    // Test with transposed tensor
                    if (input_tensor.dim() >= 2) {
                        torch::Tensor transposed = input_tensor.transpose(0, 1);
                        int64_t trans_numel = torch::numel(transposed);
                        if (trans_numel != num_elements) {
                            std::cerr << "Transposed tensor numel mismatch" << std::endl;
                        }
                    }
                    break;
                }
                case 1: {
                    // Test with contiguous tensor
                    torch::Tensor contiguous = input_tensor.contiguous();
                    int64_t cont_numel = torch::numel(contiguous);
                    if (cont_numel != num_elements) {
                        std::cerr << "Contiguous tensor numel mismatch" << std::endl;
                    }
                    break;
                }
                case 2: {
                    // Test with cloned tensor
                    torch::Tensor cloned = input_tensor.clone();
                    int64_t clone_numel = torch::numel(cloned);
                    if (clone_numel != num_elements) {
                        std::cerr << "Cloned tensor numel mismatch" << std::endl;
                    }
                    break;
                }
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