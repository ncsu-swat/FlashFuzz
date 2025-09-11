#include "fuzzer_utils.h" // General fuzzing utilities
#include <iostream>       // For cerr
#include <tuple>          // For std::get with lu_unpack result

// --- Fuzzer Entry Point ---
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    std::cout << "Start Fuzzing" << std::endl;
    try
    {
        size_t offset = 0;
        
        // Skip if we don't have enough data
        if (Size < 2) {
            return 0;
        }
        
        // Create input tensor
        torch::Tensor input = fuzzer_utils::createTensor(Data, Size, offset);
        
        // Apply torch.special functions to the tensor
        // Note: torch.special has many functions, we'll try several of them
        
        // Try torch::special::erf
        torch::Tensor result_erf = torch::special::erf(input);
        
        // Try torch::special::erfc
        torch::Tensor result_erfc = torch::special::erfc(input);
        
        // Try torch::special::erfinv
        torch::Tensor result_erfinv = torch::special::erfinv(input);
        
        // Try torch::special::expit
        torch::Tensor result_expit = torch::special::expit(input);
        
        // Try torch::special::exp2
        torch::Tensor result_exp2 = torch::special::exp2(input);
        
        // Try torch::special::gammaln
        torch::Tensor result_gammaln = torch::special::gammaln(input);
        
        // Try torch::special::digamma
        torch::Tensor result_digamma = torch::special::digamma(input);
        
        // Try torch::special::psi
        torch::Tensor result_psi = torch::special::psi(input);
        
        // Try torch::special::i0
        torch::Tensor result_i0 = torch::special::i0(input);
        
        // Try torch::special::i0e
        torch::Tensor result_i0e = torch::special::i0e(input);
        
        // Try torch::special::i1
        torch::Tensor result_i1 = torch::special::i1(input);
        
        // Try torch::special::i1e
        torch::Tensor result_i1e = torch::special::i1e(input);
        
        // Try torch::special::logit
        torch::Tensor result_logit = torch::special::logit(input);
        
        // Try torch::special::sinc
        torch::Tensor result_sinc = torch::special::sinc(input);
        
        // Try torch::special::round
        torch::Tensor result_round = torch::special::round(input);
        
        // Try torch::special::log1p
        torch::Tensor result_log1p = torch::special::log1p(input);
        
        // Try torch::log_softmax if tensor has at least 1 dimension
        if (input.dim() > 0) {
            int64_t dim = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dim = dim % input.dim();
            }
            torch::Tensor result_log_softmax = torch::log_softmax(input, dim, std::nullopt);
        }
        
        // Try torch::softmax if tensor has at least 1 dimension
        if (input.dim() > 0) {
            int64_t dim = 0;
            if (offset + sizeof(int64_t) <= Size) {
                std::memcpy(&dim, Data + offset, sizeof(int64_t));
                offset += sizeof(int64_t);
                dim = dim % input.dim();
            }
            torch::Tensor result_softmax = torch::softmax(input, dim, std::nullopt);
        }
        
        // Try torch::special::entr
        torch::Tensor result_entr = torch::special::entr(input);
        
        // Try torch::special::ndtri
        torch::Tensor result_ndtri = torch::special::ndtri(input);
        
        // Try torch::special::multigammaln if we have enough data
        if (offset + 1 <= Size) {
            int64_t p = static_cast<int64_t>(Data[offset++]) % 5 + 1; // p between 1 and 5
            torch::Tensor result_multigammaln = torch::special::multigammaln(input, p);
        }
        
        // Try torch::special::polygamma if we have enough data
        if (offset + 1 <= Size) {
            int64_t n = static_cast<int64_t>(Data[offset++]) % 5; // n between 0 and 4
            torch::Tensor result_polygamma = torch::special::polygamma(n, input);
        }
        
        // Try torch::special::zeta if we have enough data
        if (offset + 1 <= Size) {
            // Create a second tensor for the other parameter
            torch::Tensor other = fuzzer_utils::createTensor(Data, Size, offset);
            torch::Tensor result_zeta = torch::special::zeta(input, other);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1; // discard the input
    }
    return 0; // keep the input
}
