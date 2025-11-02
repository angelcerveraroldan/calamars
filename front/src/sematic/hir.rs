use calamars_core::{
    MaybeErr,
    ids::{self, ExpressionId, IdentId, SymbolId},
};

use crate::{
    sematic::{error::SemanticError, types::TypeId},
    source::FileId,
    syntax::span::Span,
};

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

impl Type {
    pub fn function_input(&self) -> &Box<[ids::TypeId]> {
        if let Type::Function { input, .. } = self {
            return input;
        }
        unreachable!("Make sure to only call this on functions!")
    }
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
    /// A function parameter, this will have no body, as it will be used when type cecking a
    /// function, not during the call
    Parameter,
    Extern,
    /// A function that has yet to have a body attached to it
    FunctionUndeclared,
    Function {
        body: ids::ExpressionId,
    },
    /// A variable that has yet to have a body attached to it
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

    pub fn update_body(&mut self, body: ids::ExpressionId) {
        let sk = match self.kind {
            SymbolKind::Parameter | SymbolKind::Extern => return (),
            SymbolKind::FunctionUndeclared | SymbolKind::Function { .. } => {
                SymbolKind::Function { body }
            }
            SymbolKind::VariableUndeclared { mutable } | SymbolKind::Variable { mutable, .. } => {
                SymbolKind::Variable { mutable, body }
            }
        };
        self.kind = sk;
    }

    pub fn ident_id(&self) -> ids::IdentId {
        self.name
    }

    pub fn name_span(&self) -> Span {
        self.name_span
    }
}

/// All of the HIR information for a module.
///
/// To have full infromation about the module, you need access to the global context, containing:
/// - TypeArena
/// - ConstantStringArena
/// - IdentArena,
/// - SymbolArena
struct Module {
    /// For now, ModuleId is the same as FileId, as each file is exactly a module.
    pub id: FileId,
    pub name: ids::IdentId,

    /// All of the symbols in this module.
    symbols: Box<[ids::SymbolId]>,
    /// All of the expressions in this module.
    expressions: Box<[ids::ExpressionId]>,

    /// Top-level public declarations.
    ///
    /// This can be used by other modules when importing this one.
    export: hashbrown::HashMap<ids::IdentId, ids::SymbolId>,
    diag: Vec<SemanticError>,
}
