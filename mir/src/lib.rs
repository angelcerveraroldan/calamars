//! An intermediate representation for the Calamars programming language
//!
//! This shuold be a stable, sugar free represenatation ready to be lowered to
//! some other IR such as cranelift.
//!
//!
//! When designing this, inspiration was taken from (among other things) LLVM and Cranelift.
//!
//! Some structures will contain direct references to part of the docs, but here are the "general"
//! links that can be used to find a lot of information:
//! LLVM: https://releases.llvm.org/18.1.4/docs/LangRef.html

use std::fmt::Debug;

pub struct FunctionId(usize);
pub struct BlockId(usize);
pub struct ValueId(usize);
pub struct DataId(usize);
pub struct TypeId(usize);

/// Ways in which we can call a function
pub enum Callee {
    /// Call a function with a given function id. These are functions defined in the same language
    Function(FunctionId),
    /// Externed functions
    Extern(String),
}

/// Basic types for the language
#[derive(Debug, Clone)]
pub enum Types {
    Unit,

    I32,
    Bool,
    Char,
    // Later:
    // - Tuples
    // - Array (fixed size)
    //
    // - Struct
    // - Enum
}

/// Constants that are known at compile time
#[derive(Debug, Clone)]
pub enum Consts {
    Unit,
    I32(i32),
    Bool(bool),
    Char(char),
}

/// Store some binary data
pub struct DataSeg {
    id: DataId,
    name: String,
    bytes: Vec<u8>,
    align: u32,
}

pub enum UnaryOperator {
    /// Negate a boolean value
    Not,
    /// The `-` in `-2`. Works for numerical types.
    Negate,
}

pub enum BinaryOperator {
    Add,
    Sub,
    Times,
    Div,
    Modulo,

    EqEq,
    NotEqual,
    Greater,
    Geq,
    Lesser,
    Leq,
}

/// Bitwise Binary operators.
///
/// TODO: There are many [missing operators](https://releases.llvm.org/18.1.4/docs/LangRef.html#bitwiseops)
/// that may be worth implementing later.
pub enum BitwiseBinaryOperator {
    And,
    Xor,
    Or,
}

pub enum InstructionKind {
    Constant(Consts),
    ConstDataPointer {
        data: DataId,
    },
    Binary {
        op: BinaryOperator,
        lhs: ValueId,
        rhs: ValueId,
    },
    BitwiseBinary {
        op: BitwiseBinaryOperator,
        lhs: ValueId,
        rhs: ValueId,
    },
    Unary {
        op: UnaryOperator,
        on: ValueId,
    },
    /// Call some function
    Call {
        callee: Callee,
        args: Vec<ValueId>,
        return_ty: TypeId,
    },
}

/// A single instruction that produces a basic type.
///
/// Reference: https://releases.llvm.org/18.1.4/docs/LangRef.html#instruction-reference
pub struct Instruct {
    dst: Option<ValueId>,
    kind: InstructionKind,
    span: Span,
}

/// Every basic block will end with a terminator instruction.
pub enum Terminator {
    /// When a return instruction is executed, control flow will return back to the calling
    /// functions context.
    Return(Option<ValueId>),
    /// Break out of a block
    Br {
        target: BlockId,
        args: Box<[ValueId]>,
    },
    BrIf {
        /// This is the result value of the predicate in the if statement
        condition: ValueId,
        /// What block to jump to if the predicate was true
        then_target: (BlockId, Box<ValueId>),
        /// What block to jump to if the predicate was false
        else_target: (BlockId, Box<ValueId>),
    },
    // Switch { .. }
}

pub struct Span {
    start: usize,
    end: usize,
}

/// Where in the source code did this come from
///
/// Since there may have been de-sugaring, this may not just be a single span
pub enum Origin {
    /// No desugaring happened, we know that this is exactly in this part of
    /// the source
    Exact { span: Span },
}

/// A [Basic Block](https://en.wikipedia.org/wiki/Basic_block)
pub struct BBlock {
    id: BlockId,
    params: Vec<(ValueId, TypeId)>,
    instructs: Vec<Instruct>,
    finally: Terminator,
}

pub struct Function {
    id: FunctionId,
    name: String,
    return_ty: TypeId,
    /// The input parameters are the output params of this block
    entry: BlockId,
    blocks: Vec<BBlock>,
}

pub struct Module {
    data: Vec<DataId>,
    funcs: Vec<FunctionId>,
}
