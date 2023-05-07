//
//  codec.cpp
//  experiment
//
//  Created by timothee.ewart on 5/6/20.
//

#include "ilc/include/codec.h"
#include "ilc/utils/utils.h"

#include <filesystem>

namespace ilc {

codec::codec(const std::string &model_path) {

    control_tf_log();
    model_path_ = std::filesystem::path(model_path);
    const auto models = model_path_builder(model_path_);

    std::for_each(models.begin(), models.end(), [&](const auto &it) {
        const std::string model_path = it.first.native() + "/";
        const std::string model_name = it.second;
        encoder_.add_model(model_path, model_name);
        decoder_.add_model(model_path, model_name);
    });
}

void codec::encode(const std::string &input, const std::string &output) {
    prepare_files(encoder_, input, output); // prepare input
    encoder_();                             // execute the TBB encoder
}

void codec::encode(const std::string &input_dir, const std::string &output_dir, const std::string &model_name) {
    model(model_name);
    encode(input_dir, output_dir);
}

void codec::decode(const std::string &input, const std::string &output) {
    prepare_files(decoder_, input, output); // prepare input
    decoder_();                             // execute the TBB decoder
}

void codec::model(const std::string model) {
    std::filesystem::path path = model_path_ / std::filesystem::path(model);
    if (!std::filesystem::is_directory(path))
        throw std::runtime_error("Invalid model.\n\n\t- Could not find " + path.string());
    // change model
    encoder_.model() = make_sha256(model_path_.string(), model);
}

void codec::ntokens(const std::size_t tokens) {
    encoder_.ntokens() = tokens;
    decoder_.ntokens() = tokens;
}

void codec::gpu(const std::size_t gpu) {
    ngpu_ = gpu;
    encoder_.add_gpu_devices(gpu);
    decoder_.add_gpu_devices(gpu);
}

void codec::tiling(const bool btiling) { encoder_.tiling() = btiling; }

void codec::canvas(const bool bcanvas) { encoder_.canvas() = bcanvas; }

void codec::set_png_compression_lvl(const int l) {
    const int low = 0;  // no compression
    const int high = 9; // max compression
    if ((l - high) * (l - low) <= 0)
        decoder_.lvl_compression() = l;
    else {
        std::string message("Zlib compression factor ");
        message += std::to_string(l) + std::string(" is not supported.\n");
        throw std::runtime_error(message.c_str());
    }
}

const int codec::get_png_compression_lvl() const { return decoder_.lvl_compression(); }
// encoder/decoder has the same number of GPU is setup
const int codec::get_number_gpu() const { return ngpu_; }
// number of tokens
const int codec::get_number_tokens() const { return encoder_.ntokens(); }
} // namespace ilc
