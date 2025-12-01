//! Kernel operation dispatch functions
//!
//! Each module provides call_* functions similar to candle-metal-kernels

pub mod binary;
pub mod matvec;
pub mod unary;

pub use binary::{
    call_add, call_binary, call_binary_strided, call_div, call_mul, BinaryOp, BinaryParams,
    TensorLayout,
};
pub use matvec::call_matvec;
pub use unary::{
    call_exp, call_gelu, call_relu, call_silu, call_unary_simple, call_unary_strided,
    UnaryOp, UnaryStridedParams,
};
