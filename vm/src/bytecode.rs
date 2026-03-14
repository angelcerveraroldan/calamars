use crate::{Register, values::Value};
use calamars_core::ids;
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
	/// Allocate a string to heap and save a pointer to some dst register
	ConstString { dst: Register, string_id: ids::StringId },
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
    /// Tail call return
    RetCall  { callee: ir::FunctionId, args: Box<[Register]> },
    /// Return the value in some register
    Ret      { src: Register },
    Phi      { dst: Register, incoming: Box<[(ir::BlockId, Register)]> },
	/// For debugging purposes only
	#[cfg(feature = "logs")] DbgPrint { dst: Register }
}
