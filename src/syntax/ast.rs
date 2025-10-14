use crate::syntax::{span::Span, token::Token};
use chumsky::{
    error::Rich,
    extra::{self, Full},
    input::{MapExtra, ValueInput},
    span::SimpleSpan,
};

pub trait TokenInput<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
impl<'a, I> TokenInput<'a> for I where I: ValueInput<'a, Token = Token, Span = SimpleSpan> {}

pub type ParserErr<'a> = extra::Err<Rich<'a, Token>>;

/// An identifier
#[derive(Debug, Clone, PartialEq)]
pub struct Ident {
    /// The identifier
    ident: String,
    /// Where it is found in the source
    pub span: Span,
}

impl Ident {
    pub fn new(ident: String, span: Span) -> Self {
        Self { ident, span }
    }
}

/// A Calamars module / file
#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub(crate) items: Vec<ClItem>,
}

/// Any one thing in the Cl language
#[derive(Debug, Clone, PartialEq)]
pub enum ClItem {
    Declaration(ClDeclaration),
    Expression(ClExpression),

    // TODO:
    Import,
}

/// Calamars Base Type kind and value
#[derive(Debug, Clone, PartialEq)]
pub enum ClLiteralKind {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    Char(char),
    Array(Vec<ClLiteral>),
}

/// Calamars Base Type, along with its span in the source
#[derive(Debug, Clone, PartialEq)]
pub struct ClLiteral {
    kind: ClLiteralKind,
    pub span: Span,
}

impl ClLiteral {
    pub fn new(kind: ClLiteralKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl From<ClLiteral> for ClExpressionKind {
    fn from(value: ClLiteral) -> Self {
        ClExpressionKind::Literal(value)
    }
}

/// Types for Calamars
#[derive(Debug, Clone, PartialEq)]
pub enum ClType {
    /// Basic / standard types such as Int, String, Char, Real, ...
    /// as well as types that require many segments, such as people.Person
    Path { segments: Vec<Ident> },
    /// An array of some type such as [Int]
    Array { elem_type: Box<Self> },
    /// A function (I1, I2, I3, ...) -> (O1, O2, O3, ...)
    Func {
        inputs: Vec<Self>,
        output: Vec<Self>,
    },
}

/// All types of possible expressions
#[derive(Debug, Clone, PartialEq)]
pub enum ClExpressionKind {
    Literal(ClLiteral),
    Identifier(Ident),

    UnaryOp(ClUnaryOp),
    BinaryOp(ClBinaryOp),

    IfStm(IfStm),
    FunctionCall(FuncCall),

    Block(ClCompoundExpression),
}

/// An expression along with its span
#[derive(Debug, Clone, PartialEq)]
pub struct ClExpression {
    kind: ClExpressionKind,
    span: Span,
}

impl ClExpression {
    pub fn new(kind: ClExpressionKind, span: Span) -> Self {
        Self { kind, span }
    }

    pub fn from_expk(kind: ClExpressionKind) -> Option<Self> {
        match kind.clone() {
            ClExpressionKind::Literal(lit) => {
                let span = lit.span;
                Some(Self::new(kind, span))
            }
            ClExpressionKind::Identifier(id) => {
                let span = id.span;
                Some(Self::new(kind, span))
            }
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Neg,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,      // +
    Sub,      // -
    Times,    // *
    Pow,      // ^
    Div,      // /
    Concat,   // ++
    Geq,      // >=
    Leq,      // <=
    EqEq,     // ==
    NotEqual, // !=

    Or,  // or
    Xor, // xor
    And, // and
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClUnaryOp {
    operator: UnaryOperator,
    on: Box<ClExpression>,
}

impl From<ClUnaryOp> for ClExpressionKind {
    fn from(value: ClUnaryOp) -> Self {
        ClExpressionKind::UnaryOp(value)
    }
}

impl ClUnaryOp {
    pub fn new(operator: UnaryOperator, on: Box<ClExpression>) -> Self {
        Self { operator, on }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClBinaryOp {
    operator: BinaryOperator,
    left: Box<ClExpression>,
    right: Box<ClExpression>,
}

impl From<ClBinaryOp> for ClExpressionKind {
    fn from(value: ClBinaryOp) -> Self {
        ClExpressionKind::BinaryOp(value)
    }
}

impl ClBinaryOp {
    pub fn new(
        operator: BinaryOperator,
        left: Box<ClExpression>,
        right: Box<ClExpression>,
    ) -> Self {
        Self {
            operator,
            left,
            right,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct IfStm {
    predicate: Box<ClExpression>,
    then: Box<ClExpression>,
    otherwise: Box<ClExpression>,
}

impl IfStm {
    pub fn new(
        predicate: Box<ClExpression>,
        then: Box<ClExpression>,
        otherwise: Box<ClExpression>,
    ) -> Self {
        Self {
            predicate,
            then,
            otherwise,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FuncCall {
    func_name: Ident,
    params: Vec<ClExpression>,
}

impl FuncCall {
    pub fn new(func_name: Ident, params: Vec<ClExpression>) -> Self {
        Self { func_name, params }
    }
}

/// An expression in the form
///
/// {
///    (<cl_item>)*
///    (<cl_expression>)?
/// }
#[derive(Debug, Clone, PartialEq)]
pub struct ClCompoundExpression {
    items: Vec<ClItem>,
    final_expr: Option<Box<ClExpression>>,
}

impl ClCompoundExpression {
    pub fn new(items: Vec<ClItem>, final_expr: Option<Box<ClExpression>>) -> Self {
        Self { items, final_expr }
    }
}

// DECLARATIONS

#[derive(Debug, Clone, PartialEq)]
pub enum ClDeclaration {
    Binding(ClBinding),
    Function(ClFuncDec),
}

/// Value and Variable declaration
#[derive(Debug, Clone, PartialEq)]
pub struct ClBinding {
    vname: Ident,
    vtype: ClType,
    assigned: Box<ClExpression>,
    mutable: bool,
}

impl ClBinding {
    pub fn new(vname: Ident, vtype: ClType, assigned: Box<ClExpression>, mutable: bool) -> Self {
        Self {
            vname,
            vtype,
            assigned,
            mutable,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClFuncDec {
    fname: Ident,
    inputs: Vec<(Ident, ClType)>,
    out_type: ClType,
    body: ClExpression,
}

impl ClFuncDec {
    pub fn new(
        fname: Ident,
        inputs: Vec<(Ident, ClType)>,
        out_type: ClType,
        body: ClExpression,
    ) -> Self {
        Self {
            fname,
            inputs,
            out_type,
            body,
        }
    }
}
