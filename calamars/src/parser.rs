//! Parser for Calamars

use std::fmt::Binary;

use chumsky::{
    input::ValueInput,
    pratt::{infix, left, prefix, right},
    prelude::*,
};

use crate::token::Token;

pub trait TokenInput<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
impl<'a, I> TokenInput<'a> for I where I: ValueInput<'a, Token = Token, Span = SimpleSpan> {}

type Ident = String;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ClExpression {
    Literal(ClLiteral),
    Identifier(Ident),

    UnaryOp(ClUnaryOp),
    BinaryOp(ClBinaryOp),

    IfStm(IfStm),
    FunctionCall(FuncCall),
}

impl From<ClLiteral> for ClExpression {
    fn from(value: ClLiteral) -> Self {
        ClExpression::Literal(value)
    }
}

impl From<ClBinaryOp> for ClExpression {
    fn from(value: ClBinaryOp) -> Self {
        ClExpression::BinaryOp(value)
    }
}

impl From<ClUnaryOp> for ClExpression {
    fn from(value: ClUnaryOp) -> Self {
        ClExpression::UnaryOp(value)
    }
}

/// Calamars Base Type Instance
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ClLiteral {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    Char(char),
    Array(Vec<Self>),
}

/// Parse any base value, including nested arrays
pub fn parse_literal<'a, I>() -> impl Parser<'a, I, ClLiteral, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    recursive(|value| {
        let atom = select! {
            Token::True      => ClLiteral::Boolean(true),
            Token::False     => ClLiteral::Boolean(false),
            Token::Int(i)    => ClLiteral::Integer(i),
            Token::Float(f)  => ClLiteral::Real(f),
            Token::String(s) => ClLiteral::String(s),
            Token::Char(c)   => ClLiteral::Char(c),
        };

        let arr = value
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket));

        choice((atom, arr.map(ClLiteral::Array)))
    })
}

/// Types for Calamars
#[derive(Debug, Clone, PartialEq, PartialOrd)]
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

impl ClType {
    pub fn new_func((from, to): (Vec<Self>, Vec<Self>)) -> Self {
        Self::Func {
            inputs: from,
            output: to,
        }
    }

    pub fn new_arr(t: Self) -> Self {
        Self::Array {
            elem_type: Box::new(t),
        }
    }

    pub fn new_path(p: Vec<Ident>) -> Self {
        Self::Path { segments: p }
    }
}

fn parse_cltype_path<'a, I>() -> impl Parser<'a, I, ClType, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    // Parse just an ident
    let ident_p = select! { Token::Ident(x)  => x };

    ident_p
        .clone()
        .then(just(Token::Dot).ignore_then(ident_p).repeated().collect())
        .map(|(head, tail): (String, Vec<String>)| {
            let mut tmp = Vec::with_capacity(tail.len() + 1);
            tmp.push(head);
            tmp.extend(tail);
            ClType::Path { segments: tmp }
        })
}

// TODO: Maybe its better to have (x, y, z) be a Tuple type, then functions are always A->B, where
// A and B are standard types ...
fn parse_cltype_annotation<'a, I>()
-> impl Parser<'a, I, ClType, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    recursive(|rec| {
        // Parse array type
        let array_type = rec
            .clone()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(ClType::new_arr);

        let params_many = rec
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<ClType>>()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        let params_one = parse_cltype_path().map(|z| vec![z]);

        // Funcitons may have one input
        let params = params_one.or(params_many);

        let function_type = params
            .clone()
            .then_ignore(just(Token::Arrow))
            .then(params)
            .map(ClType::new_func);

        function_type.or(parse_cltype_path()).or(array_type)
    })
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum UnaryOperator {
    Neg,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
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

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ClUnaryOp {
    operator: UnaryOperator,
    on: Box<ClExpression>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ClBinaryOp {
    operator: BinaryOperator,
    left: Box<ClExpression>,
    right: Box<ClExpression>,
}

/// Parse an atom, or a bracketed expression
fn parse_atom_expr_or_bracketed<'a, I>(
    expr: impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone,
) -> impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    let literal = parse_literal().map(ClExpression::Literal);
    let ident = select! { Token::Ident(s) => s }.map(ClExpression::Identifier);
    let bracket_expr = expr.delimited_by(just(Token::LParen), just(Token::RParen));
    choice((literal, ident, bracket_expr))
}

macro_rules! infix_shortcut {
    ($order:expr, $from:expr, $to:expr) => {
        infix($order, just($from), |lhs, _, rhs, _| {
            ClExpression::BinaryOp(ClBinaryOp {
                operator: $to,
                left: Box::new(lhs),
                right: Box::new(rhs),
            })
        })
    };
}

/// Parse a unary or a binary opearation
fn parse_binary_unary_ops<'a, I>(
    expr: impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone,
) -> impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    let atom = parse_atom_expr_or_bracketed(expr.clone());

    atom.pratt((
        // Not prefix
        prefix(6, just(Token::Not), |_, rhs, _| {
            ClExpression::UnaryOp(ClUnaryOp {
                operator: UnaryOperator::Neg,
                on: Box::new(rhs),
            })
        }),
        // Infix
        infix_shortcut!(right(5), Token::Pow, BinaryOperator::Pow),
        infix_shortcut!(left(4), Token::Star, BinaryOperator::Times),
        infix_shortcut!(left(4), Token::Slash, BinaryOperator::Div),
        infix_shortcut!(left(3), Token::Plus, BinaryOperator::Add),
        infix_shortcut!(left(3), Token::Minus, BinaryOperator::Sub),
        infix_shortcut!(left(3), Token::Concat, BinaryOperator::Concat),
        // Comparison
        infix_shortcut!(left(2), Token::EqualEqual, BinaryOperator::EqEq),
        infix_shortcut!(left(2), Token::NotEqual, BinaryOperator::NotEqual),
        infix_shortcut!(left(2), Token::LessEqual, BinaryOperator::Leq),
        infix_shortcut!(left(2), Token::GreaterEqual, BinaryOperator::Geq),
        infix_shortcut!(left(1), Token::And, BinaryOperator::And),
        infix_shortcut!(left(1), Token::Xor, BinaryOperator::Xor),
        infix_shortcut!(left(1), Token::Or, BinaryOperator::Or),
    ))
}

fn parse_identifier<'a, I>() -> impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    select! { Token::Ident(s) => s }
        .map(ClExpression::Identifier)
        .labelled("identifier")
}

/// Value and Variable declaration
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ClBinding {
    vname: Ident,
    vtype: ClType,
    assigned: Box<ClExpression>,
    mutable: bool,
}

fn parse_binding<'a, I>(
    expr: impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone,
) -> impl Parser<'a, I, ClBinding, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    let var_or_val = (just(Token::Var).map(|_| true)).or(just(Token::Val).map(|_| false));

    var_or_val
        .then(select! {Token::Ident(valname) => valname})
        .then_ignore(just(Token::Colon))
        .then(parse_cltype_annotation())
        .then_ignore(just(Token::Equal))
        .then(expr)
        .then_ignore(just(Token::Semicolon))
        .map(|(((mutable, vname), vtype), assigned)| ClBinding {
            vname,
            vtype,
            assigned: Box::new(assigned),
            mutable,
        })
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct IfStm {
    predicate: Box<ClExpression>,
    then: Box<ClExpression>,
    otherwise: Box<ClExpression>,
}

fn parse_if<'a, I>(
    expr: impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone,
) -> impl Parser<'a, I, IfStm, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    just(Token::If)
        .ignore_then(expr.clone())
        .then_ignore(just(Token::Then))
        .then(expr.clone())
        .then_ignore(just(Token::Else))
        .then(expr.clone())
        .map(|((e1, e2), e3)| IfStm {
            predicate: Box::new(e1),
            then: Box::new(e2),
            otherwise: Box::new(e3),
        })
        .labelled("if stm")
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct FuncCall {
    func_name: Ident,
    params: Vec<ClExpression>,
}

fn parse_function_call<'a, I>(
    expr: impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone,
) -> impl Parser<'a, I, FuncCall, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    let parse_args = expr
        .clone()
        .separated_by(just(Token::Comma))
        .collect::<Vec<ClExpression>>()
        .delimited_by(just(Token::LParen), just(Token::RParen));

    select! { Token::Ident(s) => s }
        .then_ignore(just(Token::LParen))
        .then(expr.separated_by(just(Token::Comma)).collect())
        .then_ignore(just(Token::RParen))
        .map(|(func_name, params)| FuncCall { func_name, params })
}

pub fn parse_expression<'a, I>()
-> impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    recursive(|rec| {
        let bracketed_expr = rec
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        choice((
            parse_function_call(rec.clone()).map(ClExpression::FunctionCall),
            parse_binary_unary_ops(rec.clone()),
            parse_if(rec.clone()).map(ClExpression::IfStm),
            parse_literal().map(ClExpression::from),
            parse_identifier(),
            bracketed_expr,
        ))
    })
}
