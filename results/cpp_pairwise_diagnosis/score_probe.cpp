// Read-only diagnostic harness: includes the unchanged frozen implementation.
#define main frozen_click_sampler_main
#include "../click-prestructural-v2/click_sampler.cpp"
#undef main

int main(int argc, char** argv) {
    try {
        const auto option = parse_options(argc, argv);
        const auto values = read_csv(option.input, 2*option.modes, 2*option.modes);
        Matrix covariance(2*option.modes,2*option.modes);
        for (int i=0;i<2*option.modes;++i)
            for (int j=0;j<2*option.modes;++j)
                covariance(i,j)=values[i*2*option.modes+j];
        Context context;
        Sampler sampler(option,std::move(covariance),context);
        const State state(sampler.base);
        std::cout << std::setprecision(17) << "{\"mi\":[";
        for (int i=0;i<option.modes;++i) {
            if(i)std::cout << ',';
            std::cout << '[';
            for (int j=0;j<option.modes;++j) {
                if(j)std::cout << ',';
                std::cout << sampler.pairwise_mi(i,j);
            }
            std::cout << ']';
        }
        std::cout << "],\"vacuum_singles\":[";
        for (int i=0;i<option.modes;++i) {
            if(i)std::cout << ',';
            std::cout << vacuum(state,std::vector<int>{i},1,sampler.setup_counts,KernelRole::setup);
        }
        std::cout << "],\"vacuum_pairs\":[";
        for (int i=0;i<option.modes;++i) {
            if(i)std::cout << ',';
            std::cout << '[';
            for (int j=0;j<option.modes;++j) {
                if(j)std::cout << ',';
                if(i==j)std::cout << 1;
                else std::cout << vacuum(state,std::vector<int>{std::min(i,j),std::max(i,j)},3,
                                           sampler.setup_counts,KernelRole::setup);
            }
            std::cout << ']';
        }
        std::cout << "]}\n";
    } catch(std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
