//! An intermediate representation for the Calamars programming language
//!
//! This should be a stable, sugar free representation ready to be lowered to
//! some other IR such as cranelift.
//!
//!
//! When designing this, inspiration was taken from (among other things) LLVM and Cranelift.
//!
//! Some structures will contain direct references to part of the docs, but here are the "general"
//! links that can be used to find a lot of information:
//! LLVM: https://releases.llvm.org/18.1.4/docs/LangRef.html

pub mod errors;
pub mod lower;
mod optimizations;
pub mod printer;

use calamars_core::{UncheckedArena, ids};
use front::syntax::span::Span;

use crate::{
    errors::MirErrors,
    lower::MirRes,
    optimizations::{OptFunction, PhiReturnOptimization, TailCallOptimization},
};

#[derive(Copy, Debug, Clone, PartialEq, Eq)]
pub struct FunctionId(usize);

impl From<usize> for FunctionId {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl calamars_core::Identifier for FunctionId {
    fn inner_id(&self) -> usize {
        self.0
    }
}

/// A local identifier for a `BBlock`
///
/// BlockId(0) is the first block of the working function
#[derive(Copy, Debug, PartialEq, Eq, Clone)]
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
///
/// This identifier is local, that is to say ValueId(0) is the first instruction in the working
/// function.
#[derive(Copy, Debug, Clone, Hash, PartialEq, Eq)]
pub struct ValueId(usize);

impl ValueId {
    pub fn inner(&self) -> usize {
        self.0
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataId(usize);

/// Ways in which we can call a function
#[derive(Debug, Clone, PartialEq, Eq)]
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

    I64,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Consts {
    Unit,
    I64(i64),
    Bool(bool),
    String(ids::StringId),
}

/// Store some binary data
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataSeg {
    id: DataId,
    name: String,
    bytes: Vec<u8>,
    align: u32,
}

#[derive(Debug, PartialEq, Eq)]
pub enum UnaryOperator {
    /// Negate a boolean value
    Not,
    /// The `-` in `-2`. Works for numerical types.
    Negate,
}

#[derive(Debug, PartialEq, Eq)]
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
    And,
    Or,
    Xor,
}

/// Bitwise Binary operators.
///
/// TODO: There are many [missing operators](https://releases.llvm.org/18.1.4/docs/LangRef.html#bitwiseops)
/// that may be worth implementing later.
#[derive(Debug, PartialEq, Eq)]
pub enum BitwiseBinaryOperator {
    And,
    Xor,
    Or,
}

#[derive(Debug, PartialEq, Eq)]
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
    Phi {
        ty: ids::TypeId,
        incoming: Box<[(BlockId, ValueId)]>,
    },

    Parameter {
        index: u16,
        ty: ids::TypeId,
    },
}

impl VInstructionKind {
    pub fn call_to_terminator(&self) -> MirRes<Terminator> {
        match self {
            VInstructionKind::Call {
                callee,
                args,
                return_ty,
            } => Ok(Terminator::Call {
                callee: callee.clone(),
                args: args.clone(),
                return_ty: return_ty.clone(),
            }),
            _ => Err(MirErrors::LoweringErr {
                msg: "Can only convert function calls".to_string(),
            }),
        }
    }
}

/// A single value producing instruction.
///
/// Reference: https://releases.llvm.org/18.1.4/docs/LangRef.html#instruction-reference
#[derive(Debug, PartialEq, Eq)]
pub struct VInstruct {
    pub kind: VInstructionKind,
}

/// Every basic block will end with a terminator instruction.
#[derive(Debug)]
pub enum Terminator {
    /// When a return instruction is executed, control flow will return back to the calling
    /// functions context.
    Return(Option<ValueId>),
    /// Return a function call. This is better than VInstructionKind::Call followed by return,
    /// since we know that we can use this for tail call optimization.
    Call {
        callee: Callee,
        args: Vec<ValueId>,
        return_ty: ids::TypeId,
    },
    /// Break out of a block
    Br { target: BlockId },
    BrIf {
        /// This is the result value of the predicate in the if statement
        condition: ValueId,
        /// What block to jump to if the predicate was true
        then_target: BlockId,
        /// What block to jump to if the predicate was false
        else_target: BlockId,
    },
    // Switch { .. }
}

/// Where in the source code did this come from
///
/// Since there may have been de-sugaring, this may not just be a single span
pub enum Origin {
    /// No de-sugaring happened, we know that this is exactly in this part of
    /// the source
    Exact { span: Span },
}

/// A [Basic Block](https://en.wikipedia.org/wiki/Basic_block)
#[derive(Debug, Default)]
pub struct BBlock {
    pub instructs: Vec<ValueId>,
    pub finally: Option<Terminator>,
}

impl BBlock {
    pub fn with_term(&mut self, term: Terminator) -> &mut BBlock {
        debug_assert!(self.finally.is_none(), "Cannot double-assign a terminator");

        self.finally = Some(term);
        self
    }

    pub fn with_instruct(&mut self, instruct: ValueId) -> &mut BBlock {
        debug_assert!(
            self.finally.is_none(),
            "Cannot add instructions after terminator"
        );
        self.instructs.push(instruct);
        self
    }
}

pub struct Function {
    pub name: ids::IdentId,
    pub id: FunctionId,
    pub return_ty: ids::TypeId,
    pub params: Vec<ValueId>,

    pub instructions: Vec<VInstruct>,
    pub blocks: Vec<BBlock>,
}

impl Function {
    pub fn arity(&self) -> u16 {
        self.params.len() as u16
    }
}

pub struct Module {
    pub function_arena: UncheckedArena<Function, FunctionId>,
}

impl Module {
    pub fn new(function_arena: UncheckedArena<Function, FunctionId>) -> Self {
        let mut raw = Self { function_arena };
        raw.optimize();
        raw
    }

    pub fn tco(&mut self) {
        for function in self.function_arena.inner_mut() {
            let mut tco = TailCallOptimization::new();
            if let Err(error) = tco.optimize(function, 1) {
                eprint!("{:?}", error);
            }
        }
    }

    pub fn remove_uneccesary_phis(&mut self) {
        for function in self.function_arena.inner_mut() {
            let mut pro = PhiReturnOptimization::new();
            if let Err(error) = pro.optimize(function, 1) {
                eprint!("{:?}", error);
            }
        }
    }

    /// Optimize MIR. ORDER OF THE FUNCTIONS IS CRUCIAL
    pub fn optimize(&mut self) {
        self.remove_uneccesary_phis();
        self.tco();
    }
}
