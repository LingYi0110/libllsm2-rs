use crate::*;
use llsm2::{SynthesisOptions, analyze, synthesize, synthesize_with_effect};

/// Borrowed UTF-8, NUL-terminated last failure on this thread. Successful calls
/// retain it; it expires on the next failure or thread exit. Never free it.
#[unsafe(no_mangle)]
pub extern "C" fn llsm2_last_error_message() -> *const c_char {
    catch_unwind(|| {
        LAST_ERROR.with(|v| {
            let v = v.borrow();
            if v.is_empty() {
                c"".as_ptr()
            } else {
                v.as_ptr().cast()
            }
        })
    })
    .unwrap_or(c"error message unavailable".as_ptr())
}
/// Defaults borrow a static 2/4/8 kHz axis. Adjust for low-rate input.
/// # Safety
/// out is writable option storage. See the crate's pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_analysis_options_default(
    out: *mut LlsmAnalysisOptions,
) -> LlsmStatus {
    guard(|| unsafe {
        let o = llsm2::AnalysisOptions::default();
        put(
            out,
            LlsmAnalysisOptions {
                thop: o.thop,
                max_nhar: o.max_nhar,
                max_nhar_env: o.max_nhar_env,
                npsd: o.npsd,
                chan_freqs: DEFAULT_CHANNELS.as_ptr(),
                num_chan_freqs: 3,
                lip_radius: o.lip_radius,
                refine_f0: 1,
                hm_method: LLSM_CHIRP_Z,
                rel_winsize: o.rel_winsize,
            },
        )?;
        Ok(())
    })
}
/// Default model config for source fs; static channel boundaries at or above
/// source Nyquist are omitted.
/// # Safety
/// out is writable config storage. See the crate's pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_model_config_default(
    fs: f64,
    out: *mut LlsmModelConfig,
) -> LlsmStatus {
    guard(|| unsafe {
        let mut options = llsm2::AnalysisOptions::default();
        options.chan_freqs.retain(|&f| f < fs / 2.0);
        let c = options.to_config(fs)?;
        let mut value = LlsmModelConfig::view(&c);
        value.chan_freqs = DEFAULT_CHANNELS.as_ptr();
        put(out, value)
    })
}
/// # Safety
/// out is writable option storage. See the crate's pointer contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_synthesis_options_default(
    fs: f64,
    out: *mut LlsmSynthesisOptions,
) -> LlsmStatus {
    guard(|| unsafe {
        let o = SynthesisOptions::new(fs);
        let (a, b) = o.iczt_threshold.unwrap();
        put(
            out,
            LlsmSynthesisOptions {
                fs,
                mode: LLSM_LAYER0,
                use_iczt: 1,
                iczt_a: a,
                iczt_b: b,
                random_seed: o.random_seed,
            },
        )?;
        Ok(())
    })
}
/// Analyze mono input; returned model owns all retained data.
/// # Safety
/// Input arrays/options are live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_analyze(
    options: *const LlsmAnalysisOptions,
    x: *const f64,
    nx: usize,
    fs: f64,
    f0: *const f64,
    nframes: usize,
    out: *mut *mut LlsmChunk,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            let o = arg(options)?.owned()?;
            Ok(LlsmChunk {
                inner: analyze(&o, input_slice(x, nx)?, fs, input_slice(f0, nframes)?)?.chunk,
            })
        })
    })
}
/// Create unvoiced frames; configs with nspec receive Rd=1 on each frame.
/// # Safety
/// config and its arrays are live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_create(
    config: *const LlsmModelConfig,
    nframes: usize,
    out: *mut *mut LlsmChunk,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            let c = arg(config)?.owned()?;
            c.validate()?;
            let mut inner = Chunk::new(c, nframes);
            if inner.config.nspec.is_some() {
                for f in &mut inner.frames {
                    f.rd = Some(1.0);
                }
            }
            Ok(LlsmChunk { inner })
        })
    })
}
/// Destroy and null the owned slot. An already-null handle is a no-op.
/// # Safety
/// slot owns a chunk returned here; no borrowed views remain in use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_destroy(slot: *mut *mut LlsmChunk) -> LlsmStatus {
    guard(|| unsafe { destroy(slot) })
}
/// Config borrows the chunk's channel array until mutation/destruction.
/// # Safety
/// chunk is live; out is disjoint writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_config(
    chunk: *const LlsmChunk,
    out: *mut LlsmModelConfig,
) -> LlsmStatus {
    guard(|| unsafe {
        put(out, LlsmModelConfig::view(&arg(chunk)?.inner.config))?;
        Ok(())
    })
}
/// # Safety
/// chunk is live; out is writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_frame_count(
    chunk: *const LlsmChunk,
    out: *mut usize,
) -> LlsmStatus {
    guard(|| unsafe {
        put(out, arg(chunk)?.inner.num_frames())?;
        Ok(())
    })
}
/// Return an independent owned frame copy.
/// # Safety
/// chunk is live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_get_frame(
    chunk: *const LlsmChunk,
    index: usize,
    out: *mut *mut LlsmFrame,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            Ok(LlsmFrame::new(
                arg(chunk)?
                    .inner
                    .frames
                    .get(index)
                    .ok_or(Error::Input("frame index out of range"))?
                    .clone(),
            ))
        })
    })
}
/// Copy and validate a frame against the explicitly selected layer.
/// # Safety
/// chunk is exclusively borrowed; frame is live and independent of chunk.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_set_frame(
    chunk: *mut LlsmChunk,
    index: usize,
    frame: *const LlsmFrame,
    layer: u32,
) -> LlsmStatus {
    guard(|| unsafe {
        let chunk = &mut arg_mut(chunk)?.inner;
        if index >= chunk.frames.len() {
            return Err(Error::Input("frame index out of range"));
        }
        chunk.frames[index] =
            validate_frame(chunk.config.clone(), arg(frame)?.inner.clone(), layer)?;
        Ok(())
    })
}
/// # Safety
/// chunk is a live, exclusively borrowed handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_to_layer1(chunk: *mut LlsmChunk, nfft: usize) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(chunk)?.inner.to_layer1(nfft)?;
        Ok(())
    })
}
/// # Safety
/// chunk is a live, exclusively borrowed handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_to_layer0(chunk: *mut LlsmChunk) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(chunk)?.inner.to_layer0()?;
        Ok(())
    })
}
/// # Safety
/// chunk is live and exclusively borrowed; its current phase data is valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_phase_propagate(
    chunk: *mut LlsmChunk,
    value: i32,
) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(chunk)?.inner.phase_propagate(direction(value)?);
        Ok(())
    })
}
/// # Safety
/// chunk is live and exclusively borrowed; layer1_based is 0 or 1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_phase_sync_rps(
    chunk: *mut LlsmChunk,
    layer1_based: u32,
) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(chunk)?.inner.phase_sync_rps(flag(layer1_based)?);
        Ok(())
    })
}
/// # Safety
/// chunk is live and exclusively borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_clear_layer0(chunk: *mut LlsmChunk) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(chunk)?.inner.clear_layer0();
        Ok(())
    })
}
/// # Safety
/// chunk is live and exclusively borrowed; enabled is 0 or 1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_chunk_set_pbp(
    chunk: *mut LlsmChunk,
    index: usize,
    enabled: u32,
) -> LlsmStatus {
    guard(|| unsafe {
        let enabled = flag(enabled)?;
        arg_mut(chunk)?
            .inner
            .frames
            .get_mut(index)
            .ok_or(Error::Input("frame index out of range"))?
            .pbp_enabled = enabled;
        Ok(())
    })
}
/// Deep-copy an input view and validate it against config/layer.
/// # Safety
/// Every input array is live for this call; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_frame_create(
    config: *const LlsmModelConfig,
    layer: u32,
    view: *const LlsmFrameView,
    out: *mut *mut LlsmFrame,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            Ok(LlsmFrame::new(validate_frame(
                arg(config)?.owned()?,
                arg(view)?.owned()?,
                layer,
            )?))
        })
    })
}
/// # Safety
/// slot owns a frame returned here; no borrowed views remain in use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_frame_destroy(slot: *mut *mut LlsmFrame) -> LlsmStatus {
    guard(|| unsafe { destroy(slot) })
}
/// Read-only arrays expire on owner mutation/destruction. Do not free them.
/// # Safety
/// frame is live; out is disjoint writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_frame_view(
    frame: *const LlsmFrame,
    out: *mut LlsmFrameView,
) -> LlsmStatus {
    guard(|| unsafe {
        put(out, arg(frame)?.view())?;
        Ok(())
    })
}
/// # Safety
/// frame is exclusively borrowed; config and its arrays are live.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_frame_to_layer0(
    frame: *mut LlsmFrame,
    config: *const LlsmModelConfig,
) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(frame)?.inner.to_layer0(&arg(config)?.owned()?)?;
        Ok(())
    })
}
/// # Safety
/// frame is live and exclusively borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_frame_phase_shift(frame: *mut LlsmFrame, theta: f64) -> LlsmStatus {
    guard(|| unsafe {
        if !theta.is_finite() {
            return Err(Error::Input("phase shift must be finite"));
        }
        arg_mut(frame)?.inner.phase_shift(theta);
        Ok(())
    })
}
/// # Safety
/// frame is live and exclusively borrowed; enabled is 0 or 1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_frame_set_pbp(frame: *mut LlsmFrame, enabled: u32) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(frame)?.inner.pbp_enabled = flag(enabled)?;
        Ok(())
    })
}

/// # Safety
/// frame is live and exclusively borrowed; layer1_based is 0 or 1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_frame_phase_sync_rps(
    frame: *mut LlsmFrame,
    layer1_based: u32,
) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(frame)?.inner.phase_sync_rps(flag(layer1_based)?);
        Ok(())
    })
}
/// # Safety
/// frame is live and exclusively borrowed. Subsequent synthesis must have a
/// valid Layer 1 representation or receive a reconstructed harmonic model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_frame_clear_layer0(frame: *mut LlsmFrame) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(frame)?.inner.clear_layer0();
        Ok(())
    })
}
/// Interpolate compatible frames using the core's voicing and circular-phase
/// rules. Use backward/forward phase propagation around time resampling.
/// # Safety
/// config and both frames are live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_interpolate_frames(
    config: *const LlsmModelConfig,
    left: *const LlsmFrame,
    right: *const LlsmFrame,
    ratio: f64,
    out: *mut *mut LlsmFrame,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            Ok(LlsmFrame::new(llsm2::interpolate_frames(
                &arg(config)?.owned()?,
                &arg(left)?.inner,
                &arg(right)?.inner,
                ratio,
            )?))
        })
    })
}
/// # Safety
/// config/arrays are live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_coder_create(
    config: *const LlsmModelConfig,
    order_spec: usize,
    order_bap: usize,
    out: *mut *mut LlsmCoder,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            Ok(LlsmCoder {
                inner: llsm2::Coder::new(&arg(config)?.owned()?, order_spec, order_bap)?,
            })
        })
    })
}
/// # Safety
/// slot owns a coder returned here, with no concurrent calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_coder_destroy(slot: *mut *mut LlsmCoder) -> LlsmStatus {
    guard(|| unsafe { destroy(slot) })
}
/// # Safety
/// coder is live; out is writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_coder_feature_dim(
    coder: *const LlsmCoder,
    out: *mut usize,
) -> LlsmStatus {
    guard(|| unsafe {
        put(out, arg(coder)?.inner.feature_dim())?;
        Ok(())
    })
}
/// Write exactly feature_dim values; larger output capacities are allowed.
/// # Safety
/// handles/output are live; output is disjoint from handle data.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_coder_encode(
    coder: *const LlsmCoder,
    frame: *const LlsmFrame,
    out: *mut f64,
    capacity: usize,
) -> LlsmStatus {
    guard(|| unsafe {
        let coder = &arg(coder)?.inner;
        check_output(out, capacity)?;
        if capacity < coder.feature_dim() {
            return Err(Error::Small);
        }
        coder.encode_into(&arg(frame)?.inner, output_prefix(out, coder.feature_dim()))?;
        Ok(())
    })
}
/// # Safety
/// coder/features are live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_coder_decode_layer0(
    coder: *const LlsmCoder,
    features: *const f64,
    len: usize,
    out: *mut *mut LlsmFrame,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            Ok(LlsmFrame::new(
                arg(coder)?
                    .inner
                    .decode_layer0(input_slice(features, len)?)?,
            ))
        })
    })
}
/// # Safety
/// coder/features are live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_coder_decode_layer1(
    coder: *const LlsmCoder,
    features: *const f64,
    len: usize,
    out: *mut *mut LlsmFrame,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            Ok(LlsmFrame::new(
                arg(coder)?
                    .inner
                    .decode_layer1(input_slice(features, len)?)?,
            ))
        })
    })
}
/// # Safety
/// options/config are live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_create(
    options: *const LlsmSynthesisOptions,
    config: *const LlsmModelConfig,
    capacity: usize,
    out: *mut *mut LlsmRtSynth,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            Ok(LlsmRtSynth {
                inner: llsm2::RtSynthesizer::new(
                    arg(options)?.owned()?,
                    arg(config)?.owned()?,
                    capacity,
                )?,
            })
        })
    })
}
/// Free the engine and null its slot; accepts an already-null handle.
/// # Safety
/// slot owns an engine returned here, with no concurrent use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_destroy(slot: *mut *mut LlsmRtSynth) -> LlsmStatus {
    guard(|| unsafe { destroy(slot) })
}
/// # Safety
/// rt is live; out is writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_latency(rt: *const LlsmRtSynth, out: *mut usize) -> LlsmStatus {
    guard(|| unsafe {
        put(out, arg(rt)?.inner.latency())?;
        Ok(())
    })
}
/// # Safety
/// rt is live; out is writable storage. Do not race with feed/read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_available(rt: *const LlsmRtSynth, out: *mut usize) -> LlsmStatus {
    guard(|| unsafe {
        put(out, arg(rt)?.inner.available_samples())?;
        Ok(())
    })
}
/// BufferFull preserves engine state; read and retry the same frame.
/// # Safety
/// rt is exclusively borrowed; frame is live for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_feed(
    rt: *mut LlsmRtSynth,
    frame: *const LlsmFrame,
) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(rt)?.inner.feed(&arg(frame)?.inner)?;
        Ok(())
    })
}
/// Feed without allocating an intermediate FFI frame handle.
/// # Safety
/// rt is exclusively borrowed; chunk is live and independent of rt.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_feed_chunk_frame(
    rt: *mut LlsmRtSynth,
    chunk: *const LlsmChunk,
    index: usize,
) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(rt)?.inner.feed(
            arg(chunk)?
                .inner
                .frames
                .get(index)
                .ok_or(Error::Input("frame index out of range"))?,
        )?;
        Ok(())
    })
}
/// # Safety
/// rt/output/count are live and disjoint. Null output is allowed at capacity 0.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_read(
    rt: *mut LlsmRtSynth,
    out: *mut f64,
    capacity: usize,
    count: *mut usize,
) -> LlsmStatus {
    guard(|| unsafe {
        put(count, 0)?;
        check_output(out, capacity)?;
        let rt = &mut arg_mut(rt)?.inner;
        let n = capacity.min(rt.available_samples());
        count.write(rt.read(output_prefix(out, n)));
        Ok(())
    })
}
/// Read at most the smaller capacity, retaining both output tails.
/// # Safety
/// rt/count/both outputs are live and pairwise disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_read_decomposed(
    rt: *mut LlsmRtSynth,
    periodic: *mut f64,
    periodic_capacity: usize,
    aperiodic: *mut f64,
    aperiodic_capacity: usize,
    count: *mut usize,
) -> LlsmStatus {
    guard(|| unsafe {
        put(count, 0)?;
        check_output(periodic, periodic_capacity)?;
        check_output(aperiodic, aperiodic_capacity)?;
        let rt = &mut arg_mut(rt)?.inner;
        let n = rt
            .available_samples()
            .min(periodic_capacity)
            .min(aperiodic_capacity);
        count.write(rt.read_decomposed(output_prefix(periodic, n), output_prefix(aperiodic, n)));
        Ok(())
    })
}
/// Clear all histories while retaining allocated buffers and transform plans.
/// # Safety
/// rt is live and exclusively borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_clear(rt: *mut LlsmRtSynth) -> LlsmStatus {
    guard(|| unsafe {
        arg_mut(rt)?.inner.clear();
        Ok(())
    })
}
/// Callback arguments expire on return. Caller owns callback-state rollback.
/// # Safety
/// Handles are live; callback is non-null, never unwinds/reenters rt; user is
/// valid according to the callback's own contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_rt_feed_with_effect(
    rt: *mut LlsmRtSynth,
    frame: *const LlsmFrame,
    callback: LlsmPbpCallback,
    user: *mut c_void,
) -> LlsmStatus {
    guard(|| unsafe {
        let mut effect = Effect {
            callback: callback.ok_or(Error::Null)?,
            user,
        };
        arg_mut(rt)?
            .inner
            .feed_with_effect(&arg(frame)?.inner, &mut effect)?;
        Ok(())
    })
}
/// # Safety
/// options/chunk are live; out is a new writable handle slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_synthesize(
    options: *const LlsmSynthesisOptions,
    chunk: *const LlsmChunk,
    out: *mut *mut LlsmOutput,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            Ok(LlsmOutput {
                inner: synthesize(&arg(options)?.owned()?, &arg(chunk)?.inner)?,
            })
        })
    })
}
/// # Safety
/// Same pointer rules as synthesize; callback/user obey the effect contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_synthesize_with_effect(
    options: *const LlsmSynthesisOptions,
    chunk: *const LlsmChunk,
    callback: LlsmPbpCallback,
    user: *mut c_void,
    out: *mut *mut LlsmOutput,
) -> LlsmStatus {
    guard(|| unsafe {
        create(out, || {
            let mut effect = Effect {
                callback: callback.ok_or(Error::Null)?,
                user,
            };
            Ok(LlsmOutput {
                inner: synthesize_with_effect(
                    &arg(options)?.owned()?,
                    &arg(chunk)?.inner,
                    &mut effect,
                )?,
            })
        })
    })
}
/// Borrow output arrays until the owning output handle is destroyed.
/// # Safety
/// output is live; out is disjoint writable storage.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_output_view(
    output: *const LlsmOutput,
    out: *mut LlsmOutputView,
) -> LlsmStatus {
    guard(|| unsafe {
        let o = &arg(output)?.inner;
        put(
            out,
            LlsmOutputView {
                fs: o.fs,
                len: o.y.len(),
                y: o.y.as_ptr(),
                periodic: o.y_sin.as_ptr(),
                aperiodic: o.y_noise.as_ptr(),
            },
        )?;
        Ok(())
    })
}
/// # Safety
/// slot owns an output returned here; no borrowed views remain in use.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn llsm2_output_destroy(slot: *mut *mut LlsmOutput) -> LlsmStatus {
    guard(|| unsafe { destroy(slot) })
}
