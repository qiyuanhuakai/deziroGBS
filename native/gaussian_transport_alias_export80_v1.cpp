// Read-only integer table export for independent exact-rational audits.
#include "gaussian_pnr_transport80_v4.cpp"
API int transport80b_alias(void* plan,int label,int capacity,std::uint64_t* thresholds,int* aliases,int* metadata,char* message,int cap){try{
 PrecisionGuard guard;need(plan&&thresholds&&aliases&&metadata,"Invalid alias audit pointers");auto& p=*static_cast<TransportPlan80*>(plan);need(label>=-1&&label<p.latent,"Alias audit label out of range");auto& a=label<0?p.latent_alias:p.pairs[label];need(capacity>=a.size,"Alias audit buffer is too small");std::copy(a.thresholds.begin(),a.thresholds.end(),thresholds);std::copy(a.aliases.begin(),a.aliases.end(),aliases);metadata[0]=a.bits;metadata[1]=a.size;metadata[2]=a.categories;return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
