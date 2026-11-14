//! Errors reported at public model and signal boundaries.

/// Invalid caller-supplied configuration, waveform or model data.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LlsmError {
    #[error("output buffer has {available} free samples; next frame needs {required}")]
    BufferFull { required: usize, available: usize },
    #[error("invalid configuration {field}: {reason}")]
    InvalidConfig {
        field: &'static str,
        reason: &'static str,
    },
    #[error("invalid input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    #[error("F0 track has {actual} frames; expected {minimum}..={maximum}")]
    InvalidF0Length {
        minimum: usize,
        maximum: usize,
        actual: usize,
    },
    #[error("invalid frame {index}: {reason}")]
    InvalidFrame { index: usize, reason: &'static str },
    #[error("invalid pulse {pulse_index} in frame {frame_index}: {reason}")]
    InvalidPulse {
        frame_index: usize,
        pulse_index: usize,
        reason: &'static str,
    },
}

pub(crate) fn config(
    condition: bool,
    field: &'static str,
    reason: &'static str,
) -> Result<(), LlsmError> {
    if condition {
        Ok(())
    } else {
        Err(LlsmError::InvalidConfig { field, reason })
    }
}
