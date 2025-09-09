#include "fuzzer_utils.h"
#include <iostream>
#include <tuple>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)
{
    try
    {
        size_t offset = 0;
        
        if (Size < 10) {
            return 0;
        }

        auto tensor1 = fuzzer_utils::createTensor(Data, Size, offset);
        
        if (offset >= Size) {
            return 0;
        }
        
        auto tensor2 = fuzzer_utils::createTensor(Data, Size, offset);
        
        tensor1.contiguous();
        tensor1.is_contiguous();
        tensor1.numel();
        tensor1.dim();
        tensor1.sizes();
        tensor1.strides();
        tensor1.dtype();
        tensor1.device();
        tensor1.layout();
        tensor1.requires_grad();
        tensor1.is_leaf();
        tensor1.grad_fn();
        tensor1.storage();
        tensor1.storage_offset();
        tensor1.itemsize();
        tensor1.element_size();
        tensor1.nbytes();
        tensor1.ndimension();
        tensor1.size(0);
        
        if (tensor1.dim() > 0) {
            tensor1.stride(0);
        }
        
        tensor1.is_cuda();
        tensor1.is_sparse();
        tensor1.is_mkldnn();
        tensor1.is_quantized();
        tensor1.is_meta();
        tensor1.is_complex();
        tensor1.is_floating_point();
        tensor1.is_signed();
        tensor1.has_names();
        tensor1.is_pinned();
        
        auto cloned = tensor1.clone();
        auto detached = tensor1.detach();
        auto copied = tensor1.to(tensor1.dtype());
        
        if (tensor1.numel() > 0) {
            tensor1.data_ptr();
        }
        
        if (tensor1.dtype() == torch::kFloat || tensor1.dtype() == torch::kDouble) {
            tensor1.sum();
            tensor1.mean();
            if (tensor1.numel() > 0) {
                tensor1.min();
                tensor1.max();
            }
        }
        
        if (tensor1.dtype() == torch::kBool) {
            tensor1.any();
            tensor1.all();
        }
        
        if (tensor1.sizes() == tensor2.sizes() && tensor1.dtype() == tensor2.dtype()) {
            tensor1 + tensor2;
            tensor1 - tensor2;
            tensor1 * tensor2;
            tensor1.add(tensor2);
            tensor1.sub(tensor2);
            tensor1.mul(tensor2);
            tensor1.eq(tensor2);
            tensor1.ne(tensor2);
        }
        
        if (tensor1.dim() > 1) {
            tensor1.transpose(0, 1);
            tensor1.t();
        }
        
        if (tensor1.dim() > 0) {
            tensor1.squeeze();
            tensor1.unsqueeze(0);
            tensor1.flatten();
        }
        
        tensor1.view({-1});
        tensor1.reshape({-1});
        
        if (tensor1.numel() > 1) {
            std::vector<int64_t> new_shape = {tensor1.numel()};
            tensor1.resize_(new_shape);
        }
        
        if (tensor1.dtype().isFloatingType()) {
            tensor1.abs();
            tensor1.neg();
            tensor1.sign();
            tensor1.sqrt();
            tensor1.exp();
            tensor1.log();
            tensor1.sin();
            tensor1.cos();
            tensor1.tan();
            tensor1.ceil();
            tensor1.floor();
            tensor1.round();
            tensor1.trunc();
            tensor1.frac();
        }
        
        if (tensor1.dtype().isIntegralType(false)) {
            tensor1.abs();
            tensor1.neg();
            tensor1.sign();
        }
        
        auto tensor_float = tensor1.to(torch::kFloat);
        auto tensor_double = tensor1.to(torch::kDouble);
        auto tensor_int = tensor1.to(torch::kInt32);
        auto tensor_long = tensor1.to(torch::kInt64);
        auto tensor_bool = tensor1.to(torch::kBool);
        
        if (tensor1.dim() == 2 && tensor1.size(0) == tensor1.size(1)) {
            tensor1.trace();
            tensor1.diag();
        }
        
        if (tensor1.dim() >= 2) {
            tensor1.permute({1, 0});
        }
        
        tensor1.cpu();
        
        if (tensor1.numel() == 1) {
            if (tensor1.dtype() == torch::kFloat) {
                tensor1.item<float>();
            } else if (tensor1.dtype() == torch::kDouble) {
                tensor1.item<double>();
            } else if (tensor1.dtype() == torch::kInt32) {
                tensor1.item<int32_t>();
            } else if (tensor1.dtype() == torch::kInt64) {
                tensor1.item<int64_t>();
            } else if (tensor1.dtype() == torch::kBool) {
                tensor1.item<bool>();
            }
        }
        
        if (tensor1.dim() > 0) {
            tensor1[0];
            tensor1.select(0, 0);
            tensor1.slice(0, 0, 1);
            tensor1.narrow(0, 0, 1);
            tensor1.index_select(0, torch::tensor({0}, torch::kLong));
        }
        
        if (tensor1.dim() > 1) {
            tensor1.chunk(2, 0);
            tensor1.split(1, 0);
        }
        
        torch::cat({tensor1, tensor1}, 0);
        torch::stack({tensor1, tensor1}, 0);
        
        if (tensor1.dtype().isFloatingType() && tensor1.numel() > 0) {
            tensor1.std();
            tensor1.var();
            tensor1.norm();
        }
        
        tensor1.zero_();
        tensor1.fill_(1.0);
        
        if (tensor1.dtype().isFloatingType()) {
            tensor1.uniform_(-1.0, 1.0);
            tensor1.normal_(0.0, 1.0);
        }
        
        tensor1.copy_(tensor1);
        
        auto indices = torch::nonzero(tensor1.to(torch::kBool));
        
        if (tensor1.dtype().isFloatingType()) {
            tensor1.clamp(-1.0, 1.0);
            tensor1.clamp_min(-1.0);
            tensor1.clamp_max(1.0);
        }
        
        if (tensor1.dtype().isIntegralType(false)) {
            tensor1.clamp(-10, 10);
            tensor1.clamp_min(-10);
            tensor1.clamp_max(10);
        }
        
        tensor1.masked_fill(tensor1.to(torch::kBool), 0);
        tensor1.masked_select(tensor1.to(torch::kBool));
        
        if (tensor1.dim() > 0) {
            tensor1.sort();
            tensor1.argsort();
            tensor1.topk(1);
        }
        
        if (tensor1.dtype().isFloatingType()) {
            tensor1.isnan();
            tensor1.isinf();
            tensor1.isfinite();
        }
        
        tensor1.type_as(tensor2);
        
        if (tensor1.dim() == 2) {
            tensor1.mm(tensor1.t());
        }
        
        if (tensor1.dim() >= 1) {
            tensor1.dot(tensor1.flatten());
        }
        
        tensor1.expand_as(tensor1);
        tensor1.repeat({2});
        
        if (tensor1.dim() > 0) {
            tensor1.gather(0, torch::zeros({1}, torch::kLong));
            tensor1.scatter_(0, torch::zeros({1}, torch::kLong), 1.0);
        }
        
        tensor1.backward();
        
    }
    catch (const std::exception &e)
    {
        std::cout << "Exception caught: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}