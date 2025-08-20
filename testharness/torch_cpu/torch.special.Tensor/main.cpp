#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result
#include <algorithm>      // For std::max

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input_tensor = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special operations to the tensor
        // torch.special namespace contains various special functions
        
        // Try different special functions based on remaining data
        if (offset < Size) {
            uint8_t op_selector = Data[offset++] % 10;
            
            switch (op_selector) {
                case 0:
                    torch::special::entr(input_tensor);
                    break;
                case 1:
                    torch::special::erf(input_tensor);
                    break;
                case 2:
                    torch::special::erfc(input_tensor);
                    break;
                case 3:
                    torch::special::erfcx(input_tensor);
                    break;
                case 4:
                    torch::special::erfinv(input_tensor);
                    break;
                case 5:
                    torch::special::expit(input_tensor);
                    break;
                case 6:
                    torch::special::expm1(input_tensor);
                    break;
                case 7:
                    torch::special::exp2(input_tensor);
                    break;
                case 8:
                    torch::special::gammaln(input_tensor);
                    break;
                case 9:
                    torch::special::i0(input_tensor);
                    break;
                default:
                    torch::special::log1p(input_tensor);
            }
        } else {
            // Default to log1p if no more data
            torch::special::log1p(input_tensor);
        }
        
        // Try another special function with different parameters
        if (offset + 1 < Size) {
            uint8_t op_selector = Data[offset++] % 5;
            
            switch (op_selector) {
                case 0:
                    torch::special::logit(input_tensor);
                    break;
                case 1:
                    torch::special::logsumexp(input_tensor, {0}, false);
                    break;
                case 2:
                    if (input_tensor.dim() > 0) {
                        int64_t dim = Data[offset++] % std::max(static_cast<int64_t>(1), input_tensor.dim());
                        torch::special::logsumexp(input_tensor, {dim}, false);
                    } else {
                        torch::special::logsumexp(input_tensor, {0}, false);
                    }
                    break;
                case 3:
                    torch::special::log_softmax(input_tensor, 0, c10::nullopt);
                    break;
                case 4:
                    if (input_tensor.dim() > 0) {
                        int64_t dim = Data[offset++] % std::max(static_cast<int64_t>(1), input_tensor.dim());
                        torch::special::log_softmax(input_tensor, dim, c10::nullopt);
                    } else {
                        torch::special::log_softmax(input_tensor, 0, c10::nullopt);
                    }
                    break;
                default:
                    torch::special::round(input_tensor);
            }
        }
        
        // Try a third special function
        if (offset < Size) {
            uint8_t op_selector = Data[offset++] % 5;
            
            switch (op_selector) {
                case 0:
                    torch::special::softmax(input_tensor, 0, c10::nullopt);
                    break;
                case 1:
                    if (input_tensor.dim() > 0) {
                        int64_t dim = Data[offset++] % std::max(static_cast<int64_t>(1), input_tensor.dim());
                        torch::special::softmax(input_tensor, dim, c10::nullopt);
                    } else {
                        torch::special::softmax(input_tensor, 0, c10::nullopt);
                    }
                    break;
                case 2:
                    torch::special::xlog1py(input_tensor, input_tensor);
                    break;
                case 3:
                    torch::special::xlogy(input_tensor, input_tensor);
                    break;
                case 4:
                    torch::special::zeta(input_tensor, input_tensor);
                    break;
                default:
                    torch::special::ndtr(input_tensor);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}