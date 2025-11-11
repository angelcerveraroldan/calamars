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

pub mod lower;
pub mod printer;

use calamars_core::ids;
use front::syntax::span::Span;

pub type InstructionArena = calamars_core::UncheckedArena<VInstruct, ValueId>;
pub type BlockArena = calamars_core::UncheckedArena<BBlock, BlockId>;

/// An identifier for a `BBlock`
#[derive(Copy, Debug, Clone)]
pub struct BlockId(usize);

impl From<usize> for BlockId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl calamars_core::Identifier for BlockId {
    fn inner_id(&self) -> usize {
        self.0
    }
}

/// Identifier for an SSA value
#[derive(Copy, Debug, Clone)]
pub struct ValueId(usize);

impl From<usize> for ValueId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl calamars_core::Identifier for ValueId {
    fn inner_id(&self) -> usize {
        self.0
    }
}

#[derive(Debug)]
pub struct DataId(usize);

/// Ways in which we can call a function
#[derive(Debug)]
pub enum Callee {
    /// Call a function with a given function id. These are functions defined in the same language
    Function(ids::SymbolId),
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
    I64(i64),
    Bool(bool),
    String(ids::StringId),
}

/// Store some binary data
pub struct DataSeg {
    id: DataId,
    name: String,
    bytes: Vec<u8>,
    align: u32,
}

#[derive(Debug)]
pub enum UnaryOperator {
    /// Negate a boolean value
    Not,
    /// The `-` in `-2`. Works for numerical types.
    Negate,
}

#[derive(Debug)]
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
#[derive(Debug)]
pub enum BitwiseBinaryOperator {
    And,
    Xor,
    Or,
}

#[derive(Debug)]
pub enum VInstructionKind {
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
        return_ty: ids::TypeId,
    },
}

/// A single value producing instruction.
///
/// Reference: https://releases.llvm.org/18.1.4/docs/LangRef.html#instruction-reference
pub struct VInstruct {
    dst: Option<ValueId>,
    kind: VInstructionKind,
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
    params: Vec<(ValueId, ids::TypeId)>,
    instructs: Vec<ValueId>,
    finally: Terminator,
}

pub struct Function {
    id: ids::SymbolId,
    name: ids::IdentId,
    return_ty: ids::TypeId,
    /// The input parameters are the output params of this block
    entry: BlockId,
    blocks: Vec<BlockId>,
}

pub struct Module {
    data: Vec<DataId>,
    funcs: Vec<ids::SymbolId>,
}
