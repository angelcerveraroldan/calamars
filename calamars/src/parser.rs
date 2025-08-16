//! Parser for Calamars

use chumsky::{input::ValueInput, prelude::*};

use crate::token::Token;

pub trait TokenInput<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
impl<'a, I> TokenInput<'a> for I where I: ValueInput<'a, Token = Token, Span = SimpleSpan> {}

type Ident = String;

// TODO: Many of these are not acutally expressions, will be moved later

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ClExpression {
    Literal(ClLiteral),
    Type(ClType),
    Value(ClBinding),

    // Expressions
    UnaryOp(ClUnaryOp),
    BinaryOp(ClBinaryOp),
}

impl From<ClLiteral> for ClExpression {
    fn from(value: ClLiteral) -> Self {
        ClExpression::Literal(value)
    }
}

impl From<ClType> for ClExpression {
    fn from(value: ClType) -> Self {
        ClExpression::Type(value)
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

/// Parse the base types
fn parse_atom<'a, I>() -> impl Parser<'a, I, ClLiteral, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    select! {
        Token::True      => ClLiteral::Boolean(true),
        Token::False     => ClLiteral::Boolean(false),
        Token::Int(i)    => ClLiteral::Integer(i),
        Token::Float(f)  => ClLiteral::Real(f),
        Token::String(s) => ClLiteral::String(s),
        Token::Char(c)   => ClLiteral::Char(c),
    }
}

/// Parse any base value, including nested arrays
pub fn parse_base_type<'a, I>() -> impl Parser<'a, I, ClLiteral, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    recursive(|value| {
        let arr = value
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket));

        choice((parse_atom(), arr.map(ClLiteral::Array)))
    })
}

/// Types for Calamars
///
/// TODO: Add a "total_span" to better handle errors
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

fn parse_unary_op<'a, I>() -> impl Parser<'a, I, UnaryOperator, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    select! { Token::Not => UnaryOperator::Neg }.labelled("unary operator")
}

fn parse_binary_op<'a, I>()
-> impl Parser<'a, I, BinaryOperator, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    select! {
      Token::Plus => BinaryOperator::Add,
      Token::Minus => BinaryOperator::Sub,
      Token::Star => BinaryOperator::Times,
      Token::Pow => BinaryOperator::Pow,
      Token::Slash => BinaryOperator::Div,
      Token::Concat => BinaryOperator::Concat,
      Token::GreaterEqual => BinaryOperator::Geq,
      Token::LessEqual => BinaryOperator::Leq,
      Token::EqualEqual => BinaryOperator::EqEq,
      Token::NotEqual => BinaryOperator::NotEqual,
      Token::And => BinaryOperator::And,
      Token::Or => BinaryOperator::Or,
      Token::Xor => BinaryOperator::Xor,
    }
    .labelled("binary operator")
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

fn parse_atom_expr_or_bracketed<'a, I>(
    expr: impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone,
) -> impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    let bracket_expr = expr.delimited_by(just(Token::LParen), just(Token::RParen));
    (parse_base_type().map(ClExpression::Literal)).or(bracket_expr)
}

fn parse_unary_oprator<'a, I>(
    expr: impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone,
) -> impl Parser<'a, I, ClUnaryOp, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    just(Token::Not).ignore_then(expr).map(|expr| ClUnaryOp {
        operator: UnaryOperator::Neg,
        on: Box::new(expr),
    })
}

fn parser_binary_op<'a, I>(
    expr: impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone,
) -> impl Parser<'a, I, ClBinaryOp, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    let op = select! {
      Token::Plus => BinaryOperator::Add,
      Token::Minus => BinaryOperator::Sub,
      Token::Star => BinaryOperator::Times,
      Token::Pow => BinaryOperator::Pow,
      Token::Slash => BinaryOperator::Div,
      Token::Concat => BinaryOperator::Concat,
      Token::GreaterEqual => BinaryOperator::Geq,
      Token::LessEqual => BinaryOperator::Leq,
      Token::EqualEqual => BinaryOperator::EqEq,
      Token::NotEqual => BinaryOperator::NotEqual,
      Token::And => BinaryOperator::And,
      Token::Or => BinaryOperator::Or,
      Token::Xor => BinaryOperator::Xor,
    };

    (parse_atom_expr_or_bracketed(expr.clone()))
        .then(op)
        .then(expr)
        .map(|((el, op), er)| ClBinaryOp {
            left: Box::new(el),
            right: Box::new(er),
            operator: op,
        })
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
            parser_binary_op(rec.clone()).map(ClExpression::BinaryOp),
            parse_cltype_annotation().map(ClExpression::from),
            parse_base_type().map(ClExpression::from),
            parse_binding(rec.clone()).map(ClExpression::Value),
            parse_unary_oprator(rec).map(ClExpression::UnaryOp),
            bracketed_expr,
        ))
    })
}
