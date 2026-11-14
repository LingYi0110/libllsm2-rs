//! C99 ABI with owned opaque handles and borrowed, read-only array views.
//!
//! Each returned handle is owned and must be destroyed by its matching destroy
//! function. Destroy clears the caller's slot and accepts an already-null handle.
//! Never free Rust arrays or handles with the host allocator. Views expire on
//! owner mutation/destruction. Mutation requires exclusive handle access.
//!
//! # Safety
//! Callers supply live handles of the correct type and aligned pointers to valid
//! arrays of the stated lengths. Writable outputs must be disjoint from handles,
//! inputs and each other. Null arrays are allowed only at length zero. Pointer
//! validity beyond null checks is the caller's contract. Callbacks never unwind
//! or reenter the same mutable handle. Destroy affected handles after an
//! InternalError caused by a Rust panic. panic=abort builds cannot recover.
#![deny(unsafe_op_in_unsafe_fn)]
mod exports;
mod types;
pub use exports::*;
use llsm2::{
    Chunk, GlottalFlowParams, LlsmError, ModelConfig, PbpEffect, PhaseDirection, PulseContext,
    SynthesisMode,
};
use std::{
    cell::RefCell,
    ffi::{c_char, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};
pub use types::*;

type Result<T> = std::result::Result<T, Error>;
#[derive(Debug)]
enum Error {
    Null,
    Input(&'static str),
    Small,
    Core(LlsmError),
}
impl From<LlsmError> for Error {
    fn from(e: LlsmError) -> Self {
        Self::Core(e)
    }
}
impl Error {
    fn status(&self) -> LlsmStatus {
        match self {
            Self::Null => LlsmStatus::NullPointer,
            Self::Input(_) => LlsmStatus::InvalidInput,
            Self::Small => LlsmStatus::BufferTooSmall,
            Self::Core(e) => match e {
                LlsmError::InvalidConfig { .. } => LlsmStatus::InvalidConfig,
                LlsmError::InvalidFrame { .. } => LlsmStatus::InvalidFrame,
                LlsmError::InvalidPulse { .. } => LlsmStatus::InvalidPulse,
                LlsmError::BufferFull { .. } => LlsmStatus::BufferFull,
                _ => LlsmStatus::InvalidInput,
            },
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Null => f.write_str("null pointer"),
            Self::Input(s) => f.write_str(s),
            Self::Small => f.write_str("output buffer too small"),
            Self::Core(e) => e.fmt(f),
        }
    }
}
thread_local! { static LAST_ERROR:RefCell<Vec<u8>>=const { RefCell::new(Vec::new()) }; }
fn record_error(message: &str) {
    LAST_ERROR.with(|cell| {
        let mut v = cell.borrow_mut();
        v.clear();
        v.extend_from_slice(message.as_bytes());
        v.push(0);
    });
}
fn guard(f: impl FnOnce() -> Result<()>) -> LlsmStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => LlsmStatus::Ok,
        Ok(Err(e)) => {
            record_error(&e.to_string());
            e.status()
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("Rust panic inside llsm2");
            record_error(message);
            LlsmStatus::InternalError
        }
    }
}
unsafe fn arg<'a, T>(p: *const T) -> Result<&'a T> {
    unsafe { p.as_ref() }.ok_or(Error::Null)
}
unsafe fn arg_mut<'a, T>(p: *mut T) -> Result<&'a mut T> {
    unsafe { p.as_mut() }.ok_or(Error::Null)
}
unsafe fn input_slice<'a, T>(p: *const T, n: usize) -> Result<&'a [T]> {
    if n == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(Error::Null);
    }
    if n > isize::MAX as usize / std::mem::size_of::<T>() {
        return Err(Error::Input("array length exceeds addressable size"));
    }
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}
fn check_output(p: *mut f64, n: usize) -> Result<()> {
    if n > 0 && p.is_null() {
        return Err(Error::Null);
    }
    if n > isize::MAX as usize / std::mem::size_of::<f64>() {
        return Err(Error::Input("array length exceeds addressable size"));
    }
    Ok(())
}
// The caller checked capacity and chose a prefix within it. C output storage
// may be uninitialized: initialize only that prefix before forming a Rust slice.
unsafe fn output_prefix<'a>(p: *mut f64, n: usize) -> &'a mut [f64] {
    if n == 0 {
        return &mut [];
    }
    unsafe {
        ptr::write_bytes(p, 0, n);
        std::slice::from_raw_parts_mut(p, n)
    }
}
unsafe fn put<T>(p: *mut T, value: T) -> Result<()> {
    if p.is_null() {
        return Err(Error::Null);
    }
    // Unlike &mut T, raw writes accept uninitialized caller-owned storage.
    unsafe {
        p.write(value);
    }
    Ok(())
}
unsafe fn create<T>(out: *mut *mut T, build: impl FnOnce() -> Result<T>) -> Result<()> {
    unsafe {
        put(out, ptr::null_mut())?;
        out.write(Box::into_raw(Box::new(build()?)));
    }
    Ok(())
}
unsafe fn destroy<T>(slot: *mut *mut T) -> Result<()> {
    let p = unsafe { arg_mut(slot)? };
    let value = std::mem::replace(p, ptr::null_mut());
    if !value.is_null() {
        drop(unsafe { Box::from_raw(value) });
    }
    Ok(())
}
fn flag(v: u32) -> Result<bool> {
    match v {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Error::Input("boolean value must be 0 or 1")),
    }
}
fn mode(v: u32) -> Result<SynthesisMode> {
    match v {
        0 => Ok(SynthesisMode::Layer0),
        1 => Ok(SynthesisMode::Layer1),
        _ => Err(Error::Input("invalid synthesis layer")),
    }
}
fn direction(v: i32) -> Result<PhaseDirection> {
    match v {
        1 => Ok(PhaseDirection::Forward),
        -1 => Ok(PhaseDirection::Backward),
        _ => Err(Error::Input("phase direction must be +1 or -1")),
    }
}
fn validate_frame(config: ModelConfig, frame: llsm2::Frame, layer: u32) -> Result<llsm2::Frame> {
    let mut chunk = Chunk {
        config,
        frames: vec![frame],
    };
    match mode(layer)? {
        SynthesisMode::Layer0 => chunk.validate_layer0()?,
        SynthesisMode::Layer1 => chunk.validate_layer1()?,
    }
    Ok(chunk.frames.pop().unwrap())
}
static DEFAULT_CHANNELS: [f64; 3] = [2000.0, 4000.0, 8000.0];

struct Effect {
    callback: unsafe extern "C" fn(
        *mut LlsmGlottalFlowParams,
        *const LlsmPulseContext,
        *mut c_void,
    ) -> f64,
    user: *mut c_void,
}
impl PbpEffect for Effect {
    fn modify(&mut self, g: &mut GlottalFlowParams, ctx: &PulseContext<'_>) -> f64 {
        let mut params = LlsmGlottalFlowParams {
            fa: g.fa,
            rk: g.rk,
            rg: g.rg,
            t0: g.t0,
            ee: g.ee,
        };
        let context = LlsmPulseContext {
            frame_index: ctx.frame_index,
            pulse_index: ctx.pulse_index,
            f0: ctx.frame.f0,
            rd: ctx.frame.rd.unwrap(),
        };
        let delta = unsafe { (self.callback)(&mut params, &context, self.user) };
        *g = GlottalFlowParams {
            fa: params.fa,
            rk: params.rk,
            rg: params.rg,
            t0: params.t0,
            ee: params.ee,
        };
        delta
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unwind_is_contained_and_error_messages_are_thread_local() {
        assert_eq!(
            guard(|| panic!("intentional boundary test")),
            LlsmStatus::InternalError
        );
        let message = unsafe { std::ffi::CStr::from_ptr(llsm2_last_error_message()) }
            .to_str()
            .unwrap();
        assert_eq!(message, "intentional boundary test");
        std::thread::spawn(|| {
            assert_eq!(
                unsafe { std::ffi::CStr::from_ptr(llsm2_last_error_message()) }.to_bytes(),
                b""
            );
        })
        .join()
        .unwrap();
    }
    #[test]
    fn owned_handle_destruction_drops_once_and_nulls_the_slot() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct Owned;
        impl Drop for Owned {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut slot = std::mem::MaybeUninit::uninit();
        unsafe {
            create(slot.as_mut_ptr(), || Ok(Owned)).unwrap();
            let mut handle = slot.assume_init();
            destroy(&mut handle).unwrap();
            assert!(handle.is_null());
            destroy(&mut handle).unwrap();
        }
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
    }
}
