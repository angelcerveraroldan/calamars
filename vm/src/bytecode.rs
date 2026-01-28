use crate::{Register, Value};
use ir;

#[rustfmt::skip]
#[derive(Clone, Copy, Debug)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Eq, Ne, Lt, Le, Gt, Ge,
    And, Or, Xor,
}

#[rustfmt::skip]
#[derive(Clone, Copy, Debug)]
pub enum UnOp { Not, Neg }

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Bytecode {
    /// Save constant to some register
    Const { dst: Register, k: Value },
    /// binary operation
    Bin   { op: BinOp, dst: Register, a: Register, b: Register },
    /// unary operation
    Un    { op: UnOp,  dst: Register, x: Register },
    /// break to another block
    Br    { target: ir::BlockId },
    /// conditional break to another block
    BrIf  { cond: Register, then_t: ir::BlockId, else_t: ir::BlockId },
    /// call some function
    Call  { callee: ir::FunctionId, args: Box<[Register]>, dst: Register },
    /// return the value in some register
    Ret   { src: Register },
    Phi   { dst: Register, incoming: Box<[(ir::BlockId, Register)]> },
}
