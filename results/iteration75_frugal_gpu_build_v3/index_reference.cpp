#include "fgbs/core/consts.hpp"
#include <fstream>
#include <string>
template<std::size_t N> void dump(const std::string& p){std::ofstream f(p+"_"+std::to_string(N)+".bin",std::ios::binary);const auto& a=fgbs::gbs_index_arrays<N>;f.write(reinterpret_cast<const char*>(a.data()),sizeof(a));}
int main(int argc,char** argv){if(argc!=2)return 2;dump<192>(argv[1]);dump<288>(argv[1]);dump<384>(argv[1]);}
