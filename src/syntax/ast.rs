use crate::syntax::token::Token;
use chumsky::{error::Rich, extra, input::ValueInput, span::SimpleSpan};

pub trait TokenInput<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
impl<'a, I> TokenInput<'a> for I where I: ValueInput<'a, Token = Token, Span = SimpleSpan> {}

pub type ParserErr<'a> = extra::Err<Rich<'a, Token>>;
pub type Ident = String;

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

/// Calamars Base Type Instance
#[derive(Debug, Clone, PartialEq)]
pub enum ClLiteral {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    Char(char),
    Array(Vec<Self>),
}

impl From<ClLiteral> for ClExpression {
    fn from(value: ClLiteral) -> Self {
        ClExpression::Literal(value)
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

// Expressions

#[derive(Debug, Clone, PartialEq)]
pub enum ClExpression {
    Literal(ClLiteral),
    Identifier(Ident),

    UnaryOp(ClUnaryOp),
    BinaryOp(ClBinaryOp),

    IfStm(IfStm),
    FunctionCall(FuncCall),

    Block(ClCompoundExpression),
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

impl From<ClUnaryOp> for ClExpression {
    fn from(value: ClUnaryOp) -> Self {
        ClExpression::UnaryOp(value)
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

impl From<ClBinaryOp> for ClExpression {
    fn from(value: ClBinaryOp) -> Self {
        ClExpression::BinaryOp(value)
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
