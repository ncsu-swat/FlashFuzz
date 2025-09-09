#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least some data to work with
        if (Size < 16) {
            return 0;
        }
        
        // Extract tensor creation parameters
        uint8_t tensor_type = extract_uint8(Data, Size, offset);
        uint8_t num_dims = extract_uint8(Data, Size, offset) % 6 + 1; // 1-6 dimensions
        
        // Create shape vector
        std::vector<int64_t> shape;
        for (int i = 0; i < num_dims; i++) {
            int64_t dim_size = extract_uint16(Data, Size, offset) % 100 + 1; // 1-100 size per dim
            shape.push_back(dim_size);
        }
        
        // Test different tensor creation methods
        torch::Tensor tensor;
        
        switch (tensor_type % 8) {
            case 0: {
                // Create tensor with zeros
                tensor = torch::zeros(shape);
                break;
            }
            case 1: {
                // Create tensor with ones
                tensor = torch::ones(shape);
                break;
            }
            case 2: {
                // Create tensor with random values
                tensor = torch::randn(shape);
                break;
            }
            case 3: {
                // Create tensor with uniform random values
                tensor = torch::rand(shape);
                break;
            }
            case 4: {
                // Create tensor filled with a specific value
                float fill_value = extract_float(Data, Size, offset);
                tensor = torch::full(shape, fill_value);
                break;
            }
            case 5: {
                // Create tensor from data
                tensor = torch::randn(shape);
                break;
            }
            case 6: {
                // Create empty tensor
                tensor = torch::empty(shape);
                break;
            }
            case 7: {
                // Create tensor with arange-like data
                int64_t total_elements = 1;
                for (auto s : shape) total_elements *= s;
                tensor = torch::arange(0, total_elements).reshape(shape);
                break;
            }
        }
        
        // Test basic tensor operations
        if (offset < Size) {
            uint8_t op_type = extract_uint8(Data, Size, offset);
            
            switch (op_type % 10) {
                case 0: {
                    // Test tensor properties
                    auto sizes = tensor.sizes();
                    auto strides = tensor.strides();
                    auto dtype = tensor.dtype();
                    auto device = tensor.device();
                    auto numel = tensor.numel();
                    auto ndim = tensor.ndimension();
                    break;
                }
                case 1: {
                    // Test tensor reshaping
                    if (tensor.numel() > 0) {
                        auto reshaped = tensor.reshape({-1});
                        auto flattened = tensor.flatten();
                    }
                    break;
                }
                case 2: {
                    // Test tensor cloning and copying
                    auto cloned = tensor.clone();
                    auto copied = tensor.detach();
                    break;
                }
                case 3: {
                    // Test tensor type conversions
                    if (offset + 1 < Size) {
                        uint8_t dtype_choice = extract_uint8(Data, Size, offset) % 6;
                        switch (dtype_choice) {
                            case 0: tensor = tensor.to(torch::kFloat32); break;
                            case 1: tensor = tensor.to(torch::kFloat64); break;
                            case 2: tensor = tensor.to(torch::kInt32); break;
                            case 3: tensor = tensor.to(torch::kInt64); break;
                            case 4: tensor = tensor.to(torch::kBool); break;
                            case 5: tensor = tensor.to(torch::kUInt8); break;
                        }
                    }
                    break;
                }
                case 4: {
                    // Test tensor indexing (if tensor has elements)
                    if (tensor.numel() > 0 && tensor.ndimension() > 0) {
                        auto first_elem = tensor.flatten()[0];
                        if (tensor.ndimension() == 1 && tensor.size(0) > 0) {
                            auto slice = tensor.slice(0, 0, std::min(tensor.size(0), int64_t(5)));
                        }
                    }
                    break;
                }
                case 5: {
                    // Test tensor arithmetic operations
                    if (tensor.numel() > 0) {
                        float scalar = extract_float(Data, Size, offset);
                        auto added = tensor + scalar;
                        auto multiplied = tensor * scalar;
                        auto subtracted = tensor - scalar;
                        if (scalar != 0.0f) {
                            auto divided = tensor / scalar;
                        }
                    }
                    break;
                }
                case 6: {
                    // Test tensor reduction operations
                    if (tensor.numel() > 0) {
                        auto sum_val = tensor.sum();
                        auto mean_val = tensor.mean();
                        auto max_val = tensor.max();
                        auto min_val = tensor.min();
                    }
                    break;
                }
                case 7: {
                    // Test tensor comparison operations
                    if (tensor.numel() > 0) {
                        float threshold = extract_float(Data, Size, offset);
                        auto gt_mask = tensor > threshold;
                        auto lt_mask = tensor < threshold;
                        auto eq_mask = tensor == threshold;
                    }
                    break;
                }
                case 8: {
                    // Test tensor mathematical functions
                    if (tensor.numel() > 0 && tensor.dtype().isFloatingPoint()) {
                        auto abs_tensor = torch::abs(tensor);
                        auto sqrt_tensor = torch::sqrt(torch::abs(tensor) + 1e-8);
                        auto exp_tensor = torch::exp(torch::clamp(tensor, -10, 10));
                        auto log_tensor = torch::log(torch::abs(tensor) + 1e-8);
                    }
                    break;
                }
                case 9: {
                    // Test tensor concatenation and stacking
                    if (tensor.numel() > 0) {
                        auto tensor2 = torch::ones_like(tensor);
                        if (tensor.ndimension() > 0) {
                            auto concatenated = torch::cat({tensor, tensor2}, 0);
                            auto stacked = torch::stack({tensor, tensor2}, 0);
                        }
                    }
                    break;
                }
            }
        }
        
        // Test tensor with different devices if CUDA is available
        if (offset < Size && torch::cuda::is_available()) {
            uint8_t device_test = extract_uint8(Data, Size, offset);
            if (device_test % 2 == 0) {
                try {
                    auto cuda_tensor = tensor.to(torch::kCUDA);
                    auto back_to_cpu = cuda_tensor.to(torch::kCPU);
                } catch (...) {
                    // CUDA operations might fail, ignore
                }
            }
        }
        
        // Test tensor serialization/deserialization concepts
        if (offset < Size) {
            uint8_t serialize_test = extract_uint8(Data, Size, offset);
            if (serialize_test % 2 == 0 && tensor.numel() > 0) {
                // Test tensor data access
                if (tensor.dtype() == torch::kFloat32) {
                    auto data_ptr = tensor.data_ptr<float>();
                    if (data_ptr && tensor.numel() > 0) {
                        volatile float first_val = data_ptr[0]; // Prevent optimization
                    }
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