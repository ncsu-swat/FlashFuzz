#include "fuzzer_utils.h"
#include <torch/torch.h>
#include <iostream>
#include <cstring>
#include <limits>

// Helper to extract a scalar value from fuzzer data
template<typename T>
T extractScalar(const uint8_t* data, size_t& offset, size_t size) {
    T value = T(0);
    size_t bytes_needed = sizeof(T);
    if (offset + bytes_needed <= size) {
        std::memcpy(&value, data + offset, bytes_needed);
        offset += bytes_needed;
    } else if (offset < size) {
        // Partial data available
        size_t available = size - offset;
        std::memcpy(&value, data + offset, available);
        offset = size;
    }
    return value;
}

// Helper to create a scalar tensor with same dtype as input
torch::Tensor createScalarTensor(const uint8_t* data, size_t& offset, size_t size, torch::ScalarType dtype) {
    torch::Tensor scalar;
    
    switch(dtype) {
        case torch::kFloat:
            scalar = torch::tensor(extractScalar<float>(data, offset, size), torch::dtype(dtype));
            break;
        case torch::kDouble:
            scalar = torch::tensor(extractScalar<double>(data, offset, size), torch::dtype(dtype));
            break;
        case torch::kHalf:
            scalar = torch::tensor(extractScalar<float>(data, offset, size), torch::dtype(torch::kFloat)).to(torch::kHalf);
            break;
        case torch::kBFloat16:
            scalar = torch::tensor(extractScalar<float>(data, offset, size), torch::dtype(torch::kFloat)).to(torch::kBFloat16);
            break;
        case torch::kInt8:
            scalar = torch::tensor(extractScalar<int8_t>(data, offset, size), torch::dtype(dtype));
            break;
        case torch::kUInt8:
            scalar = torch::tensor(extractScalar<uint8_t>(data, offset, size), torch::dtype(dtype));
            break;
        case torch::kInt16:
            scalar = torch::tensor(extractScalar<int16_t>(data, offset, size), torch::dtype(dtype));
            break;
        case torch::kInt32:
            scalar = torch::tensor(extractScalar<int32_t>(data, offset, size), torch::dtype(dtype));
            break;
        case torch::kInt64:
            scalar = torch::tensor(extractScalar<int64_t>(data, offset, size), torch::dtype(dtype));
            break;
        case torch::kBool:
            scalar = torch::tensor(extractScalar<uint8_t>(data, offset, size) != 0, torch::dtype(dtype));
            break;
        case torch::kComplexFloat:
            {
                float real = extractScalar<float>(data, offset, size);
                float imag = extractScalar<float>(data, offset, size);
                scalar = torch::complex(torch::tensor(real), torch::tensor(imag)).to(dtype);
            }
            break;
        case torch::kComplexDouble:
            {
                double real = extractScalar<double>(data, offset, size);
                double imag = extractScalar<double>(data, offset, size);
                scalar = torch::complex(torch::tensor(real), torch::tensor(imag)).to(dtype);
            }
            break;
        default:
            scalar = torch::tensor(0.0f, torch::dtype(torch::kFloat));
            break;
    }
    
    return scalar;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        if (Size < 3) {
            // Need at least 3 bytes: 2 for tensor metadata, 1 for operation mode
            return 0;
        }

        size_t offset = 0;

        // Create the main tensor to operate on
        torch::Tensor tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Save original for comparison if needed
        torch::Tensor original = tensor.clone();
        
        // Extract operation mode
        uint8_t op_mode = 0;
        if (offset < Size) {
            op_mode = Data[offset++];
        }
        
        // Decide how to provide the min value based on op_mode
        switch (op_mode % 4) {
            case 0: {
                // Use a scalar value extracted from data
                if (tensor.scalar_type() == torch::kComplexFloat || 
                    tensor.scalar_type() == torch::kComplexDouble) {
                    // Complex types not supported for clamp_min_
                    return 0;
                }
                
                double min_val = extractScalar<double>(Data, offset, Size);
                
                // Apply various transformations to min_val to test edge cases
                uint8_t transform = 0;
                if (offset < Size) {
                    transform = Data[offset++];
                }
                
                switch (transform % 8) {
                    case 0: break; // Use as-is
                    case 1: min_val = -min_val; break;
                    case 2: min_val = std::numeric_limits<double>::infinity(); break;
                    case 3: min_val = -std::numeric_limits<double>::infinity(); break;
                    case 4: min_val = std::numeric_limits<double>::quiet_NaN(); break;
                    case 5: min_val = 0.0; break;
                    case 6: min_val = std::numeric_limits<double>::min(); break;
                    case 7: min_val = std::numeric_limits<double>::max(); break;
                }
                
                tensor.clamp_min_(min_val);
                break;
            }
            case 1: {
                // Use a tensor as min (same shape)
                if (tensor.scalar_type() == torch::kComplexFloat || 
                    tensor.scalar_type() == torch::kComplexDouble) {
                    return 0;
                }
                
                torch::Tensor min_tensor = torch::randn_like(tensor);
                
                // Modify min_tensor based on fuzzer input
                if (offset < Size) {
                    uint8_t mod = Data[offset++];
                    switch (mod % 5) {
                        case 0: break; // Random values
                        case 1: min_tensor.fill_(0); break;
                        case 2: min_tensor.fill_(std::numeric_limits<float>::infinity()); break;
                        case 3: min_tensor.fill_(-std::numeric_limits<float>::infinity()); break;
                        case 4: min_tensor = tensor.clone(); break; // Same as input
                    }
                }
                
                tensor.clamp_min_(min_tensor);
                break;
            }
            case 2: {
                // Use a scalar tensor
                if (tensor.scalar_type() == torch::kComplexFloat || 
                    tensor.scalar_type() == torch::kComplexDouble) {
                    return 0;
                }
                
                torch::Tensor scalar_min = createScalarTensor(Data, offset, Size, tensor.scalar_type());
                tensor.clamp_min_(scalar_min);
                break;
            }
            case 3: {
                // Test with broadcasted tensor
                if (tensor.scalar_type() == torch::kComplexFloat || 
                    tensor.scalar_type() == torch::kComplexDouble || 
                    tensor.dim() == 0) {
                    // Skip complex or scalar tensors for broadcasting test
                    return 0;
                }
                
                // Create a tensor that can be broadcasted
                std::vector<int64_t> broadcast_shape;
                for (int i = 0; i < tensor.dim(); ++i) {
                    if (offset < Size && Data[offset++] % 2 == 0) {
                        broadcast_shape.push_back(1);
                    } else {
                        broadcast_shape.push_back(tensor.size(i));
                    }
                }
                
                torch::Tensor broadcast_min = torch::randn(broadcast_shape, tensor.options());
                tensor.clamp_min_(broadcast_min);
                break;
            }
        }
        
        // Additional edge case testing
        if (offset < Size) {
            uint8_t extra_test = Data[offset++];
            
            switch (extra_test % 4) {
                case 0:
                    // Test on a view
                    if (tensor.numel() > 1) {
                        auto view = tensor.view({-1});
                        view.clamp_min_(0.0);
                    }
                    break;
                case 1:
                    // Test on a slice
                    if (tensor.dim() > 0 && tensor.size(0) > 1) {
                        auto slice = tensor.narrow(0, 0, 1);
                        slice.clamp_min_(1.0);
                    }
                    break;
                case 2:
                    // Test on transposed tensor
                    if (tensor.dim() >= 2) {
                        auto transposed = tensor.transpose(0, 1);
                        transposed.clamp_min_(-1.0);
                    }
                    break;
                case 3:
                    // Test on non-contiguous tensor
                    if (tensor.dim() >= 2 && tensor.size(0) > 1 && tensor.size(1) > 1) {
                        auto non_contig = tensor.select(0, 0);
                        non_contig.clamp_min_(0.5);
                    }
                    break;
            }
        }
        
        // Verify tensor is still valid
        if (tensor.defined()) {
            // Check for NaN/Inf in result (not necessarily an error, but good to know)
            if (tensor.dtype().isFloatingPoint()) {
                bool has_nan = torch::any(torch::isnan(tensor)).item<bool>();
                bool has_inf = torch::any(torch::isinf(tensor)).item<bool>();
                
                // These are valid outcomes, just noting them
                if (has_nan || has_inf) {
                    // Could log if needed, but these are valid results
                }
            }
            
            // Verify in-place operation actually modified the tensor
            // (unless min was greater than all values)
        }
        
    }
    catch (const c10::Error &e)
    {
        // PyTorch-specific errors - these are expected for invalid operations
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    catch (...)
    {
        // Unknown exception
        std::cout << "Exception caught: Unknown exception" << std::endl;
        return -1;
    }
    
    return 0; // keep the input
}