#include "fgbs/core/random_utils.hpp"
#include <climits>
#include <iostream>

int main(){
    unsigned seed;
    while(std::cin>>seed){
        fgbs::UniformRandomInt<fgbs::usize> generator(seed,0,ULLONG_MAX);
        std::cout<<fgbs::get_random_vector<unsigned>(generator,1)[0]<<'\n';
    }
}
