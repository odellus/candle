//! Error handling for HIP backend

use thiserror::Error;

#[derive(Error, Debug)]
pub enum HipError {
    #[error("HIP error: {0}")]
    Hip(String),

    #[error("Module load error: {0}")]
    ModuleLoad(String),

    #[error("Kernel launch error: {0}")]
    KernelLaunch(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, HipError>;

/// Extension trait for wrapping HIP errors
pub trait WrapErr<T> {
    fn w(self) -> crate::Result<T>;
}

impl<T> WrapErr<T> for Result<T> {
    fn w(self) -> crate::Result<T> {
        self.map_err(crate::Error::wrap)
    }
}
