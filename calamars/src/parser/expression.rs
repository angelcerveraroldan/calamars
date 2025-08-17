use chumsky::{
    pratt::{infix, left, prefix, right},
    prelude::*,
};

use crate::parser::{ClLiteral, Ident, ParserErr, TokenInput, parse_literal};
use crate::token::Token;

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
    expr: impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone,
) -> impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let literal = parse_literal()
        .map(ClExpression::Literal)
        .labelled("literal");
    let ident = select! { Token::Ident(s) => s }
        .map(ClExpression::Identifier)
        .labelled("identifier");
    let bracket_expr = expr
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .labelled("(<expr>)");
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
    expr: impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone,
) -> impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone
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
    .labelled("Binary/Unary operation")
}

fn parse_identifier<'a, I>() -> impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    select! { Token::Ident(s) => s }
        .map(ClExpression::Identifier)
        .labelled("identifier")
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct IfStm {
    predicate: Box<ClExpression>,
    then: Box<ClExpression>,
    otherwise: Box<ClExpression>,
}

fn parse_if<'a, I>(
    expr: impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone,
) -> impl Parser<'a, I, IfStm, ParserErr<'a>> + Clone
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
    expr: impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone,
) -> impl Parser<'a, I, FuncCall, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let parse_args = expr
        .clone()
        .separated_by(just(Token::Comma))
        .collect::<Vec<ClExpression>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .labelled("function arguments");

    select! { Token::Ident(s) => s }
        .then_ignore(just(Token::LParen))
        .then(expr.separated_by(just(Token::Comma)).collect())
        .then_ignore(just(Token::RParen))
        .map(|(func_name, params)| FuncCall { func_name, params })
        .labelled("function call")
}

pub fn parse_expression<'a, I>() -> impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone
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
