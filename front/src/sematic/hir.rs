use calamars_core::{
    MaybeErr,
    ids::{self, ExpressionId, IdentId},
};

use crate::{sematic::types::TypeId, syntax::span::Span};

pub type TypeArena = calamars_core::InternArena<Type, ids::TypeId>;

/// An arena for compile-time known strings.
pub type ConstantStringArena = calamars_core::InternArena<String, ids::StringId>;

/// An arena for the names of functions
pub type IdentArena = calamars_core::InternArena<String, ids::IdentId>;

pub type ExpressionArena = calamars_core::Arena<Expr, ids::ExpressionId>;

pub type SymbolArena = calamars_core::UncheckedArena<Symbol, ids::SymbolId>;

#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Type {
    /// Among other things, this is used for when we expected a type, but found ClType::None
    ///
    /// We can:
    /// 1. Try to recover (type inference)
    /// 2. Throw a detailed error
    Error,

    // Primitives
    Integer,
    Float,
    Boolean,
    String,
    Char,
    Unit,

    Array(ids::TypeId),
    Function {
        input: Box<[ids::TypeId]>,
        output: ids::TypeId,
    },
}

impl MaybeErr for Type {
    const ERR: Self = Type::Error;
}

#[derive(Debug, PartialEq)]
pub enum Const {
    I64(i64),
    Bool(bool),
    String(ids::StringId),
}

#[derive(Debug, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mult,
    Div,
    Mod,
}

#[derive(Debug, PartialEq)]
pub enum Expr {
    Err,
    Literal {
        constant: Const,
        span: Span,
    },
    /// A function or a var/val is being used, for example `x`.
    Identifier {
        /// This tells us exactly what "declaration" we are referring to
        id: ids::SymbolId,
        span: Span,
    },
    BinaryOperation {
        operator: BinOp,
        lhs: ids::ExpressionId,
        rhs: ids::ExpressionId,
        span: Span,
    },
    Call {
        /// The function may not be an identifier since we can have HOFs, or lambdas in
        /// structs, ...
        f: ids::ExpressionId,
        inputs: Box<[ids::ExpressionId]>,
        span: Span,
    },
}

impl MaybeErr for Expr {
    const ERR: Self = Expr::Err;
}

#[derive(Debug)]
pub enum SymbolKind {
    Parameter,
    Extern,
    FunctionUndeclared,
    Function {
        body: ids::ExpressionId,
    },
    VariableUndeclared {
        mutable: bool,
    },
    Variable {
        body: ids::ExpressionId,
        mutable: bool,
    },
}

#[derive(Debug)]
pub struct Symbol {
    kind: SymbolKind,
    ty: ids::TypeId,
    name: ids::IdentId,
    /// Span to the identifier for this symbol
    name_span: Span,
    /// Span of the entire declaration for this symbol
    decl_span: Span,
}

impl Symbol {
    pub fn new(
        kind: SymbolKind,
        ty: ids::TypeId,
        name: ids::IdentId,
        name_span: Span,
        decl_span: Span,
    ) -> Self {
        Self {
            kind,
            ty,
            name,
            name_span,
            decl_span,
        }
    }
}
