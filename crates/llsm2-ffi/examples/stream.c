#include "llsm2.h"
#include <math.h>
#include <stdio.h>
#include <assert.h>

#define CHECK(call) do { LlsmStatus s = (call); if (s != LLSM_STATUS_OK) { fprintf(stderr, "%s: %s\n", #call, llsm2_last_error_message()); exit(1); } } while (0)

static double attenuate(LlsmGlottalFlowParams *g, const LlsmPulseContext *ctx, void *user) {
    size_t *calls = (size_t *)user;
    assert(ctx->pulse_index == *calls);
    *calls += 1; g->ee *= 0.95; return 0.0;
}

int main(void) {
    const double fs = 44100.0;
    double x[22050], f0[172];
    for (size_t i=0; i<22050; ++i) x[i] = 0.1*cos(6.283185307179586*200.0*(double)i/fs);
    for (size_t i=0; i<172; ++i) f0[i] = 200.0;
    LlsmAnalysisOptions analysis;
    LlsmSynthesisOptions synthesis;
    LlsmModelConfig config;
    LlsmChunk *chunk = NULL, *padding = NULL;
    LlsmCoder *coder = NULL;
    LlsmFrame *frame = NULL, *decoded = NULL;
    LlsmRtSynth *rt = NULL;
    LlsmOutput *offline = NULL;
    CHECK(llsm2_analysis_options_default(&analysis));
    analysis.thop = 128.0/fs; analysis.refine_f0 = 0;
    CHECK(llsm2_analyze(&analysis,x,22050,fs,f0,172,&chunk));
    CHECK(llsm2_chunk_to_layer1(chunk,2048));
    CHECK(llsm2_chunk_phase_sync_rps(chunk,1));
    CHECK(llsm2_chunk_clear_layer0(chunk));
    for (size_t i=0; i<172; ++i) CHECK(llsm2_chunk_set_pbp(chunk,i,(uint32_t)((i/50)%2)));
    CHECK(llsm2_chunk_phase_propagate(chunk,1));
    CHECK(llsm2_chunk_config(chunk,&config));
    CHECK(llsm2_coder_create(&config,64,5,&coder));
    CHECK(llsm2_chunk_get_frame(chunk,0,&frame));
    size_t dim = 0; CHECK(llsm2_coder_feature_dim(coder,&dim)); assert(dim == 72);
    double *features = (double *)malloc(dim*sizeof(double)); assert(features);
    CHECK(llsm2_coder_encode(coder,frame,features,dim));
    CHECK(llsm2_coder_decode_layer1(coder,features,dim,&decoded));
    CHECK(llsm2_frame_to_layer0(decoded,&config));
    CHECK(llsm2_frame_phase_shift(decoded,0.1));
    CHECK(llsm2_frame_phase_sync_rps(decoded,1));
    CHECK(llsm2_frame_clear_layer0(decoded));
    LlsmFrame *midpoint = NULL;
    CHECK(llsm2_interpolate_frames(&config,frame,decoded,0.5,&midpoint));
    CHECK(llsm2_frame_destroy(&midpoint));
    free(features);
    CHECK(llsm2_synthesis_options_default(fs,&synthesis)); synthesis.mode = LLSM_LAYER1;
    CHECK(llsm2_synthesize(&synthesis,chunk,&offline));
    LlsmOutputView reference; CHECK(llsm2_output_view(offline,&reference));
    CHECK(llsm2_rt_create(&synthesis,&config,4096,&rt));
    CHECK(llsm2_chunk_create(&config,1,&padding));
    size_t delay=0,total=0,compared=0; double noise_error=0;
    CHECK(llsm2_rt_latency(rt,&delay));
    const size_t pad_frames = (delay+255)/128+4;
    for (size_t i=0; i<172+pad_frames; ++i) {
        CHECK(llsm2_rt_feed_chunk_frame(rt,i<172?chunk:padding,i<172?i:0));
        for (;;) {
            double p[67], ap[71]; size_t n=0;
            CHECK(llsm2_rt_read_decomposed(rt,p,67,ap,71,&n)); if (!n) break;
            for (size_t j=0; j<n; ++j,++total) {
                assert(isfinite(p[j]) && isfinite(ap[j]));
                if (total<delay) { assert(p[j]==0.0 && ap[j]==0.0); }
                else if (total-delay<reference.len) {
                    size_t k=total-delay; assert(fabs(p[j]-reference.periodic[k])<1e-10);
                    double d=ap[j]-reference.aperiodic[k];noise_error+=d*d;++compared;
                }
            }
        }
    }
    assert(compared==reference.len);assert(sqrt(noise_error/(double)compared)<1e-4);
    CHECK(llsm2_rt_clear(rt));size_t n=123;CHECK(llsm2_rt_available(rt,&n));assert(n==0);
    CHECK(llsm2_frame_set_pbp(frame,1));size_t calls=0;
    CHECK(llsm2_rt_feed_with_effect(rt,frame,attenuate,&calls));assert(calls>0);
    CHECK(llsm2_rt_destroy(&rt));assert(rt==NULL);CHECK(llsm2_rt_destroy(&rt));
    CHECK(llsm2_frame_destroy(&frame));CHECK(llsm2_frame_destroy(&decoded));
    CHECK(llsm2_coder_destroy(&coder));CHECK(llsm2_output_destroy(&offline));
    CHECK(llsm2_chunk_destroy(&padding));CHECK(llsm2_chunk_destroy(&chunk));
    puts("C ABI streaming, Coder, callback, clear and ownership smoke test passed.");
    return 0;
}
