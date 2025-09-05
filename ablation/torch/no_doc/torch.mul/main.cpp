#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        // Need at least a few bytes to construct meaningful inputs
        if (Size < 4) {
            return 0;
        }
        
        // Parse operation mode from first byte
        uint8_t op_mode = Data[offset++];
        
        // Determine multiplication variant based on fuzzer input
        enum MulMode {
            TENSOR_TENSOR = 0,
            TENSOR_SCALAR = 1,
            SCALAR_TENSOR = 2,
            TENSOR_TENSOR_OUT = 3,
            BROADCAST_MUL = 4,
            INPLACE_MUL = 5
        };
        
        MulMode mode = static_cast<MulMode>(op_mode % 6);
        
        // Parse whether to use alpha parameter (for torch.mul with alpha)
        bool use_alpha = (offset < Size) ? (Data[offset++] % 2 == 1) : false;
        double alpha = 1.0;
        if (use_alpha && offset + sizeof(double) <= Size) {
            std::memcpy(&alpha, Data + offset, sizeof(double));
            offset += sizeof(double);
            // Constrain alpha to reasonable range to avoid numerical issues
            alpha = std::fmod(alpha, 1000.0);
            if (std::isnan(alpha) || std::isinf(alpha)) {
                alpha = 1.0;
            }
        }
        
        try {
            switch (mode) {
                case TENSOR_TENSOR: {
                    // Create two tensors for element-wise multiplication
                    torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Try different multiplication approaches
                    torch::Tensor result = torch::mul(tensor1, tensor2);
                    
                    // Also test with alpha parameter if applicable
                    if (use_alpha) {
                        torch::Tensor result_alpha = torch::mul(tensor1, tensor2, alpha);
                    }
                    
                    // Test operator overload
                    torch::Tensor result_op = tensor1 * tensor2;
                    break;
                }
                
                case TENSOR_SCALAR: {
                    // Create tensor and parse scalar
                    torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Parse scalar value
                    double scalar = 1.0;
                    if (offset + sizeof(double) <= Size) {
                        std::memcpy(&scalar, Data + offset, sizeof(double));
                        offset += sizeof(double);
                        // Constrain scalar to avoid overflow
                        scalar = std::fmod(scalar, 1e6);
                        if (std::isnan(scalar) || std::isinf(scalar)) {
                            scalar = 1.0;
                        }
                    }
                    
                    // Test tensor-scalar multiplication
                    torch::Tensor result = torch::mul(tensor, scalar);
                    
                    // Test with Scalar object
                    torch::Scalar scalar_obj(scalar);
                    torch::Tensor result2 = torch::mul(tensor, scalar_obj);
                    
                    // Test operator overload
                    torch::Tensor result_op = tensor * scalar;
                    break;
                }
                
                case SCALAR_TENSOR: {
                    // Parse scalar and create tensor
                    double scalar = 1.0;
                    if (offset + sizeof(double) <= Size) {
                        std::memcpy(&scalar, Data + offset, sizeof(double));
                        offset += sizeof(double);
                        scalar = std::fmod(scalar, 1e6);
                        if (std::isnan(scalar) || std::isinf(scalar)) {
                            scalar = 1.0;
                        }
                    }
                    
                    torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Test scalar-tensor multiplication
                    torch::Tensor result = torch::mul(scalar, tensor);
                    
                    // Test with Scalar object
                    torch::Scalar scalar_obj(scalar);
                    torch::Tensor result2 = torch::mul(scalar_obj, tensor);
                    
                    // Test operator overload
                    torch::Tensor result_op = scalar * tensor;
                    break;
                }
                
                case TENSOR_TENSOR_OUT: {
                    // Test out-of-place variant with pre-allocated output
                    torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Create output tensor with compatible shape
                    torch::Tensor out;
                    
                    // Determine output shape (broadcasting rules)
                    try {
                        auto result_shape = torch::broadcast_shapes({tensor1.sizes().vec(), tensor2.sizes().vec()});
                        out = torch::empty(result_shape, tensor1.options());
                        torch::mul_out(out, tensor1, tensor2);
                    } catch (const c10::Error& e) {
                        // Broadcasting might fail, try with matching shapes
                        if (tensor1.numel() > 0) {
                            out = torch::empty_like(tensor1);
                            // This might still fail if shapes are incompatible
                            torch::mul_out(out, tensor1, tensor2);
                        }
                    }
                    break;
                }
                
                case BROADCAST_MUL: {
                    // Test broadcasting scenarios
                    torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Create second tensor with potentially different shape for broadcasting
                    torch::Tensor tensor2;
                    if (offset < Size && Data[offset++] % 2 == 0) {
                        // Create scalar tensor for broadcasting
                        tensor2 = torch::tensor({1.5}, tensor1.options());
                    } else {
                        tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                    }
                    
                    // Attempt multiplication with broadcasting
                    try {
                        torch::Tensor result = torch::mul(tensor1, tensor2);
                        
                        // Test broadcasting with different dimension arrangements
                        if (tensor1.dim() > 0 && tensor2.dim() > 0) {
                            // Try unsqueezing dimensions for broadcasting tests
                            torch::Tensor t1_unsqueezed = tensor1.unsqueeze(-1);
                            torch::Tensor t2_unsqueezed = tensor2.unsqueeze(0);
                            torch::Tensor broadcast_result = torch::mul(t1_unsqueezed, t2_unsqueezed);
                        }
                    } catch (const c10::Error& e) {
                        // Broadcasting might fail for incompatible shapes, which is expected
                    }
                    break;
                }
                
                case INPLACE_MUL: {
                    // Test in-place multiplication
                    torch::Tensor tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
                    torch::Tensor tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
                    
                    // Clone for in-place operation
                    torch::Tensor tensor1_copy = tensor1.clone();
                    
                    try {
                        // In-place multiplication
                        tensor1_copy.mul_(tensor2);
                        
                        // Test scalar in-place multiplication
                        double scalar = 2.5;
                        if (offset + sizeof(double) <= Size) {
                            std::memcpy(&scalar, Data + offset, sizeof(double));
                            scalar = std::fmod(scalar, 100.0);
                            if (std::isnan(scalar) || std::isinf(scalar)) {
                                scalar = 1.0;
                            }
                        }
                        torch::Tensor tensor3 = tensor1.clone();
                        tensor3.mul_(scalar);
                        
                        // Test *= operator
                        torch::Tensor tensor4 = tensor1.clone();
                        tensor4 *= tensor2;
                        
                    } catch (const c10::Error& e) {
                        // In-place operations might fail for shape mismatches
                    }
                    break;
                }
            }
            
            // Additional edge case testing with remaining data
            if (offset < Size) {
                // Test with zero-dimensional tensors (scalars)
                torch::Tensor scalar_tensor = torch::tensor(3.14, torch::dtype(torch::kFloat32));
                torch::Tensor regular_tensor = fuzzer_utils::createTensor(Data, Size, offset);
                torch::Tensor result = torch::mul(scalar_tensor, regular_tensor);
                
                // Test with empty tensors
                if (offset < Size && Data[offset] % 4 == 0) {
                    torch::Tensor empty = torch::empty({0}, regular_tensor.options());
                    torch::Tensor empty_result = torch::mul(empty, empty);
                }
                
                // Test with different devices if available
                if (torch::cuda::is_available() && offset < Size && Data[offset] % 8 == 0) {
                    torch::Tensor cpu_tensor = regular_tensor;
                    torch::Tensor cuda_tensor = regular_tensor.to(torch::kCUDA);
                    torch::Tensor cuda_result = torch::mul(cuda_tensor, cuda_tensor);
                    torch::Tensor back_to_cpu = cuda_result.to(torch::kCPU);
                }
            }
            
        } catch (const c10::Error& e) {
            // PyTorch-specific errors (shape mismatches, broadcasting failures, etc.)
            // These are expected for some input combinations
        } catch (const std::bad_alloc& e) {
            // Memory allocation failures for very large tensors
            // Continue fuzzing with other inputs
        }
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}