use crate::{op, Layout, Result};

/// Trait for operations that map a single tensor element-wise
pub trait Map1 {
    fn map(&self, s: &super::VulkanStorageSlice, device: &super::VulkanDevice, layout: &Layout) -> Result<super::VulkanStorageSlice>;
}

/// Trait for operations that map two tensors element-wise
pub trait Map2 {
    fn map(
        &self,
        lhs: &super::VulkanStorageSlice,
        rhs: &super::VulkanStorageSlice,
        device: &super::VulkanDevice,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<super::VulkanStorageSlice>;
}

// Helper struct for affine operations (ax + b)
pub struct Affine(pub f64, pub f64);

impl Map1 for Affine {
    fn map(&self, s: &super::VulkanStorageSlice, _device: &super::VulkanDevice, _layout: &Layout) -> Result<super::VulkanStorageSlice> {
        // Implementation will be added
        todo!("Affine operation")
    }
}

// Helper for binary operations
pub trait BinaryOpT {
    const NAME: &'static str;
    const OP: op::BinaryOp;
}

pub struct Add;
impl BinaryOpT for Add {
    const NAME: &'static str = "add";
    const OP: op::BinaryOp = op::BinaryOp::Add;
}

pub struct Sub;
impl BinaryOpT for Sub {
    const NAME: &'static str = "sub";
    const OP: op::BinaryOp = op::BinaryOp::Sub;
}

pub struct Mul;
impl BinaryOpT for Mul {
    const NAME: &'static str = "mul";
    const OP: op::BinaryOp = op::BinaryOp::Mul;
}

pub struct Div;
impl BinaryOpT for Div {
    const NAME: &'static str = "div";
    const OP: op::BinaryOp = op::BinaryOp::Div;
}
