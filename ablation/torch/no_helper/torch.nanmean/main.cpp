#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

// Helper to consume bytes from fuzzer input
template<typename T>
bool consumeBytes(const uint8_t*& data, size_t& size, T& out) {
    if (size < sizeof(T)) return false;
    std::memcpy(&out, data, sizeof(T));
    data += sizeof(T);
    size -= sizeof(T);
    return true;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 8) return 0;
    
    try {
        // Consume configuration bytes
        uint8_t rank, dtype_idx, keepdim, use_dim, use_out, add_nans;
        if (!consumeBytes(data, size, rank)) return 0;
        if (!consumeBytes(data, size, dtype_idx)) return 0;
        if (!consumeBytes(data, size, keepdim)) return 0;
        if (!consumeBytes(data, size, use_dim)) return 0;
        if (!consumeBytes(data, size, use_out)) return 0;
        if (!consumeBytes(data, size, add_nans)) return 0;
        
        // Limit rank to reasonable value
        rank = (rank % 5) + 1;
        
        // Build shape
        std::vector<int64_t> shape;
        for (int i = 0; i < rank; i++) {
            uint8_t dim_size;
            if (!consumeBytes(data, size, dim_size)) {
                shape.push_back(1);
            } else {
                shape.push_back((dim_size % 8) + 1);  // Keep dimensions small
            }
        }
        
        // Select dtype
        std::vector<torch::ScalarType> dtypes = {
            torch::kFloat32, torch::kFloat64, torch::kFloat16,
            torch::kInt32, torch::kInt64, torch::kInt8, torch::kUInt8
        };
        torch::ScalarType dtype = dtypes[dtype_idx % dtypes.size()];
        
        // Create tensor with random data
        torch::Tensor input;
        int64_t numel = 1;
        for (auto d : shape) numel *= d;
        
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64 || dtype == torch::kFloat16) {
            // For floating point types, create with randn and optionally add NaNs
            input = torch::randn(shape, torch::dtype(dtype));
            
            if (add_nans % 3 == 0 && numel > 0) {
                // Add some NaN values
                auto flat = input.flatten();
                int num_nans = (add_nans % numel) + 1;
                for (int i = 0; i < num_nans && i < numel; i++) {
                    uint8_t nan_idx;
                    if (consumeBytes(data, size, nan_idx)) {
                        flat[nan_idx % numel] = std::numeric_limits<float>::quiet_NaN();
                    }
                }
            }
        } else {
            // For integer types
            input = torch::randint(0, 100, shape, torch::dtype(dtype));
        }
        
        // Prepare optional dim argument
        torch::OptionalIntArrayRef dim_arg;
        std::vector<int64_t> dims;
        if (use_dim % 2 == 0) {
            uint8_t num_dims;
            if (consumeBytes(data, size, num_dims)) {
                num_dims = (num_dims % rank) + 1;
                for (int i = 0; i < num_dims; i++) {
                    uint8_t d;
                    if (consumeBytes(data, size, d)) {
                        dims.push_back(d % rank);
                    }
                }
                if (!dims.empty()) {
                    dim_arg = dims;
                }
            }
        }
        
        // Prepare optional output tensor
        torch::Tensor out;
        bool has_out = (use_out % 3 == 0);
        if (has_out && dim_arg.has_value()) {
            // Calculate output shape
            std::vector<int64_t> out_shape = shape;
            if (keepdim % 2 == 0) {
                for (auto d : dims) {
                    if (d >= 0 && d < rank) {
                        out_shape[d] = 1;
                    }
                }
            } else {
                // Remove dimensions (complex to calculate, just create empty)
                out = torch::empty({}, torch::dtype(torch::kFloat32));
            }
            if (!out.defined()) {
                out = torch::empty(out_shape, torch::dtype(torch::kFloat32));
            }
        }
        
        // Call nanmean with various configurations
        torch::Tensor result;
        
        if (has_out && out.defined()) {
            if (dim_arg.has_value()) {
                result = torch::nanmean_out(out, input, dim_arg.value(), keepdim % 2 == 0);
            } else {
                // nanmean with output but no dim (reduce all)
                result = input.nanmean();
                out.resize_as_(result);
                out.copy_(result);
            }
        } else {
            if (dim_arg.has_value()) {
                result = torch::nanmean(input, dim_arg.value(), keepdim % 2 == 0);
            } else {
                result = input.nanmean();
            }
        }
        
        // Also test with dtype conversion
        uint8_t test_dtype_conv;
        if (consumeBytes(data, size, test_dtype_conv) && test_dtype_conv % 4 == 0) {
            if (dtype != torch::kFloat64) {
                auto result2 = torch::nanmean(input.to(torch::kFloat64), dim_arg, keepdim % 2 == 0);
            }
        }
        
        // Test edge cases
        if (numel == 0) {
            // Empty tensor
            auto empty_result = torch::nanmean(torch::empty({0}));
        }
        
        // Test all-NaN tensor
        if (dtype == torch::kFloat32 || dtype == torch::kFloat64) {
            auto all_nan = torch::full(shape, std::numeric_limits<float>::quiet_NaN(), torch::dtype(dtype));
            auto nan_result = torch::nanmean(all_nan);
        }
        
    } catch (const c10::Error& e) {
        // PyTorch errors are expected for invalid operations
        return 0;
    } catch (const std::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}