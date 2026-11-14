use llsm2_ffi::*;
use std::{
    ffi::{CStr, c_void},
    mem::MaybeUninit,
    ptr,
};
fn ok(s: LlsmStatus) {
    assert_eq!(
        s,
        LlsmStatus::Ok,
        "{}",
        unsafe { CStr::from_ptr(llsm2_last_error_message()) }.to_string_lossy()
    );
}
unsafe fn defaults() -> (LlsmModelConfig, LlsmSynthesisOptions) {
    let mut c = MaybeUninit::uninit();
    let mut o = MaybeUninit::uninit();
    ok(unsafe { llsm2_model_config_default(44100.0, c.as_mut_ptr()) });
    ok(unsafe { llsm2_synthesis_options_default(44100.0, o.as_mut_ptr()) });
    unsafe { (c.assume_init(), o.assume_init()) }
}
#[test]
fn null_dimensions_modes_full_buffer_and_ownership_have_explicit_statuses() {
    unsafe {
        assert_eq!(
            llsm2_analysis_options_default(ptr::null_mut()),
            LlsmStatus::NullPointer
        );
        let (config, mut options) = defaults();
        let mut low = MaybeUninit::uninit();
        ok(llsm2_model_config_default(16000.0, low.as_mut_ptr()));
        assert_eq!(low.assume_init().num_chan_freqs, 2);
        let mut rt = ptr::null_mut();
        assert_eq!(
            llsm2_rt_create(&options, &config, 0, &mut rt),
            LlsmStatus::InvalidConfig
        );
        assert!(rt.is_null());
        options.mode = u32::MAX;
        assert_eq!(
            llsm2_rt_create(&options, &config, 4096, &mut rt),
            LlsmStatus::InvalidInput
        );
        options.mode = LLSM_LAYER0;
        assert_eq!(
            llsm2_rt_create(ptr::null(), &config, 4096, &mut rt),
            LlsmStatus::NullPointer
        );
        ok(llsm2_rt_create(&options, &config, 222, &mut rt));
        let mut chunk = ptr::null_mut();
        ok(llsm2_chunk_create(&config, 1, &mut chunk));
        ok(llsm2_rt_feed_chunk_frame(rt, chunk, 0));
        assert_eq!(
            llsm2_rt_feed_chunk_frame(rt, chunk, 0),
            LlsmStatus::BufferFull
        );
        assert!(
            CStr::from_ptr(llsm2_last_error_message())
                .to_str()
                .unwrap()
                .contains("free samples")
        );
        let mut count = 999;
        ok(llsm2_rt_read(rt, ptr::null_mut(), 0, &mut count));
        assert_eq!(count, 0);
        assert_eq!(
            llsm2_rt_read(rt, ptr::null_mut(), 1, &mut count),
            LlsmStatus::NullPointer
        );
        assert_eq!(count, 0);
        let mut p = [123.0; 7];
        let mut ap = [456.0; 3];
        ok(llsm2_rt_read_decomposed(
            rt,
            p.as_mut_ptr(),
            7,
            ap.as_mut_ptr(),
            3,
            &mut count,
        ));
        assert_eq!(count, 3);
        assert_eq!(&p[3..], &[123.0; 4]);
        let mut rest = [0.0; 256];
        ok(llsm2_rt_read(rt, rest.as_mut_ptr(), 256, &mut count));
        assert_eq!(count, 218);
        ok(llsm2_rt_feed_chunk_frame(rt, chunk, 0));
        ok(llsm2_rt_clear(rt));
        ok(llsm2_rt_available(rt, &mut count));
        assert_eq!(count, 0);
        let mut frame = ptr::null_mut();
        assert_eq!(
            llsm2_chunk_get_frame(chunk, 10, &mut frame),
            LlsmStatus::InvalidInput
        );
        assert!(frame.is_null());
        let mut bad = config;
        bad.chan_freqs = ptr::null();
        let mut failed = ptr::dangling_mut();
        assert_eq!(
            llsm2_chunk_create(&bad, 1, &mut failed),
            LlsmStatus::NullPointer
        );
        assert!(failed.is_null());
        bad.num_chan_freqs = usize::MAX;
        bad.chan_freqs = config.chan_freqs;
        assert_eq!(
            llsm2_chunk_create(&bad, 1, &mut failed),
            LlsmStatus::InvalidInput
        );
        ok(llsm2_rt_destroy(&mut rt));
        assert!(rt.is_null());
        ok(llsm2_rt_destroy(&mut rt));
        ok(llsm2_chunk_destroy(&mut chunk));
        assert!(chunk.is_null());
        assert_eq!(llsm2_rt_destroy(ptr::null_mut()), LlsmStatus::NullPointer);
    }
}

#[derive(Default)]
struct Calls {
    count: usize,
    last_frame: usize,
    bad: bool,
}
unsafe extern "C" fn effect(
    g: *mut LlsmGlottalFlowParams,
    ctx: *const LlsmPulseContext,
    user: *mut c_void,
) -> f64 {
    let (g, ctx, c) = unsafe { (&mut *g, &*ctx, &mut *user.cast::<Calls>()) };
    if ctx.pulse_index != c.count || ctx.frame_index < c.last_frame {
        c.bad = true;
    }
    c.count += 1;
    c.last_frame = ctx.frame_index;
    g.ee *= 0.9;
    0.0
}

#[test]
fn analysis_coder_views_streaming_and_effect_callbacks_match_native_rust() {
    unsafe {
        let fs = 44100.0;
        let x: Vec<_> = (0..8820)
            .map(|i| 0.1 * (std::f64::consts::TAU * 200.0 * i as f64 / fs).cos())
            .collect();
        let f0 = vec![200.0; 40];
        let mut options = MaybeUninit::uninit();
        ok(llsm2_analysis_options_default(options.as_mut_ptr()));
        let mut options = options.assume_init();
        options.refine_f0 = 0;
        let mut chunk = ptr::null_mut();
        ok(llsm2_analyze(
            &options,
            x.as_ptr(),
            x.len(),
            fs,
            f0.as_ptr(),
            f0.len(),
            &mut chunk,
        ));
        let mut n = 0;
        ok(llsm2_chunk_frame_count(chunk, &mut n));
        assert_eq!(n, 40);
        ok(llsm2_chunk_to_layer1(chunk, 256));
        ok(llsm2_chunk_phase_sync_rps(chunk, 1));
        ok(llsm2_chunk_clear_layer0(chunk));
        for i in 0..n {
            ok(llsm2_chunk_set_pbp(chunk, i, 1));
        }
        ok(llsm2_chunk_phase_propagate(chunk, 1));
        let mut c = MaybeUninit::uninit();
        ok(llsm2_chunk_config(chunk, c.as_mut_ptr()));
        let mut c = c.assume_init();
        assert_eq!(c.nspec, 129);
        let (_, mut so) = defaults();
        so.mode = LLSM_LAYER1;
        let mut output = ptr::null_mut();
        ok(llsm2_synthesize(&so, chunk, &mut output));
        let mut ov = MaybeUninit::uninit();
        ok(llsm2_output_view(output, ov.as_mut_ptr()));
        let ov = ov.assume_init();
        assert!(
            std::slice::from_raw_parts(ov.y, ov.len)
                .iter()
                .all(|x| x.is_finite())
        );
        let mut frame = ptr::null_mut();
        ok(llsm2_chunk_get_frame(chunk, 0, &mut frame));
        let mut view = MaybeUninit::uninit();
        ok(llsm2_frame_view(frame, view.as_mut_ptr()));
        let view = view.assume_init();
        assert_eq!(view.nspec, 129);
        assert_eq!(view.has_harmonic, 0);
        let mut copy = ptr::null_mut();
        ok(llsm2_frame_create(&c, LLSM_LAYER1, &view, &mut copy));
        let mut copy_view = MaybeUninit::uninit();
        ok(llsm2_frame_view(copy, copy_view.as_mut_ptr()));
        let copy_view = copy_view.assume_init();
        assert_ne!(view.vt_magn, copy_view.vt_magn);
        assert_eq!(
            std::slice::from_raw_parts(view.vt_magn, 129),
            std::slice::from_raw_parts(copy_view.vt_magn, 129)
        );
        let mut coder = ptr::null_mut();
        ok(llsm2_coder_create(&c, 40, 5, &mut coder));
        let mut dim = 0;
        ok(llsm2_coder_feature_dim(coder, &mut dim));
        assert_eq!(dim, 48);
        let mut features = vec![123.0; dim + 3];
        assert_eq!(
            llsm2_coder_encode(coder, frame, features.as_mut_ptr(), dim - 1),
            LlsmStatus::BufferTooSmall
        );
        ok(llsm2_coder_encode(
            coder,
            frame,
            features.as_mut_ptr(),
            features.len(),
        ));
        assert_eq!(&features[dim..], &[123.0; 3]);
        let mut uninitialized = [MaybeUninit::<f64>::uninit(); 48];
        ok(llsm2_coder_encode(
            coder,
            frame,
            uninitialized.as_mut_ptr().cast(),
            48,
        ));
        for (slot, &expected) in uninitialized.iter().zip(&features) {
            assert_eq!(slot.assume_init(), expected);
        }
        let mut decoded0 = ptr::null_mut();
        let mut decoded1 = ptr::null_mut();
        ok(llsm2_coder_decode_layer0(
            coder,
            features.as_ptr(),
            dim,
            &mut decoded0,
        ));
        ok(llsm2_coder_decode_layer1(
            coder,
            features.as_ptr(),
            dim,
            &mut decoded1,
        ));
        ok(llsm2_frame_to_layer0(decoded1, &c));
        ok(llsm2_frame_phase_shift(decoded1, 0.1));
        let mut holder = ptr::null_mut();
        ok(llsm2_chunk_create(&c, 1, &mut holder));
        ok(llsm2_chunk_set_frame(holder, 0, decoded0, LLSM_LAYER0));
        ok(llsm2_chunk_to_layer0(chunk)); // also exercises the in-place API
        ok(llsm2_chunk_phase_propagate(chunk, -1));
        ok(llsm2_chunk_phase_propagate(chunk, 1));
        ok(llsm2_chunk_config(chunk, &mut c));
        let mut native = llsm2::analyze(
            &llsm2::AnalysisOptions {
                refine_f0: false,
                ..llsm2::AnalysisOptions::default()
            },
            &x,
            fs,
            &f0,
        )
        .unwrap()
        .chunk;
        native.to_layer1(256).unwrap();
        native.phase_sync_rps(true);
        native.clear_layer0();
        for f in &mut native.frames {
            f.pbp_enabled = true;
        }
        native.phase_propagate(llsm2::PhaseDirection::Forward);
        let native_options = llsm2::SynthesisOptions {
            mode: llsm2::SynthesisMode::Layer1,
            ..llsm2::SynthesisOptions::new(fs)
        };
        let mut native_rt =
            llsm2::RtSynthesizer::new(native_options, native.config.clone(), 4096).unwrap();
        let mut rt = ptr::null_mut();
        ok(llsm2_rt_create(&so, &c, 4096, &mut rt));
        let mut latency = 0;
        ok(llsm2_rt_latency(rt, &mut latency));
        assert_eq!(latency, native_rt.latency());
        // These are raw output slots, not pre-initialized Rust values.
        let mut count_slot = MaybeUninit::<usize>::uninit();
        ok(llsm2_rt_available(rt, count_slot.as_mut_ptr()));
        assert_eq!(count_slot.assume_init(), 0);
        let mut calls = Calls::default();
        let mut padding = ptr::null_mut();
        ok(llsm2_chunk_create(&c, 1, &mut padding));
        let mut native_pad = llsm2::Chunk::new(native.config.clone(), 1).frames.remove(0);
        native_pad.rd = Some(1.0);
        let mut audible = false;
        for i in 0..n + 100 {
            let mut f = ptr::null_mut();
            ok(llsm2_chunk_get_frame(
                if i < n { chunk } else { padding },
                if i < n { i } else { 0 },
                &mut f,
            ));
            ok(llsm2_rt_feed_with_effect(
                rt,
                f,
                Some(effect),
                (&mut calls as *mut Calls).cast(),
            ));
            let mut e = |g: &mut llsm2::GlottalFlowParams, _: &llsm2::PulseContext<'_>| {
                g.ee *= 0.9;
                0.0
            };
            native_rt
                .feed_with_effect(
                    if i < n {
                        &native.frames[i]
                    } else {
                        &native_pad
                    },
                    &mut e,
                )
                .unwrap();
            loop {
                let mut a = [0.0; 127];
                let mut b = [0.0; 127];
                let mut count = 0;
                ok(llsm2_rt_read(rt, a.as_mut_ptr(), 127, &mut count));
                let expected = native_rt.read(&mut b);
                assert_eq!(count, expected);
                for (a, b) in a[..count].iter().zip(&b) {
                    audible |= a.abs() > 1e-6;
                    assert!((a - b).abs() < 1e-12);
                }
                if count == 0 {
                    break;
                }
            }
            ok(llsm2_frame_destroy(&mut f));
        }
        assert!(audible);
        assert!(calls.count > 10 && !calls.bad);
        ok(llsm2_rt_clear(rt));
        calls = Calls::default();
        ok(llsm2_rt_feed_with_effect(
            rt,
            frame,
            Some(effect),
            (&mut calls as *mut Calls).cast(),
        ));
        assert!(!calls.bad);
        let mut raw_audio = [MaybeUninit::<f64>::uninit(); 256];
        let mut count_slot = MaybeUninit::<usize>::uninit();
        ok(llsm2_rt_read(
            rt,
            raw_audio.as_mut_ptr().cast(),
            256,
            count_slot.as_mut_ptr(),
        ));
        let used = count_slot.assume_init();
        assert!(used > 0);
        assert!(
            raw_audio[..used]
                .iter()
                .all(|v| v.assume_init().is_finite())
        );
        ok(llsm2_frame_destroy(&mut frame));
        let mut valid_after_source_drop = MaybeUninit::uninit();
        ok(llsm2_frame_view(copy, valid_after_source_drop.as_mut_ptr()));
        assert_eq!(valid_after_source_drop.assume_init().f0, 200.0);
        ok(llsm2_rt_destroy(&mut rt));
        ok(llsm2_chunk_destroy(&mut padding));
        ok(llsm2_output_destroy(&mut output));
        ok(llsm2_coder_destroy(&mut coder));
        for slot in [&mut copy, &mut decoded0, &mut decoded1] {
            ok(llsm2_frame_destroy(slot));
        }
        ok(llsm2_chunk_destroy(&mut chunk));
        ok(llsm2_chunk_destroy(&mut holder));
    }
}
