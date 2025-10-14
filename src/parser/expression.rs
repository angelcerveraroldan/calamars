use chumsky::{
    pratt::{infix, left, prefix, right},
    prelude::*,
};

use crate::syntax::ast::*;
use crate::{
    parser::{parse_cl_item, parse_literal},
    syntax::token::Token,
};

/// Prse an atom, or a bracketed expression
fn parse_atom_expr_or_bracketed<'a, I>(
    expr: impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone,
) -> impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let literal = parse_literal()
        .map(ClExpressionKind::Literal)
        .map(|l| ClExpression::from_expk(l).unwrap())
        .labelled("literal");
    let ident = parse_identifier()
        .map(ClExpressionKind::Identifier)
        .map(|l| ClExpression::from_expk(l).unwrap())
        .labelled("identifier");
    let bracket_expr = expr
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .labelled("(<expr>)");
    choice((literal, ident, bracket_expr))
}

macro_rules! infix_shortcut {
    ($order:expr, $from:expr, $to:expr) => {
        infix($order, just($from), |lhs, _, rhs, extra| {
            let expk =
                ClExpressionKind::BinaryOp(ClBinaryOp::new($to, Box::new(lhs), Box::new(rhs)));
            ClExpression::new(expk, extra.span())
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
        prefix(6, just(Token::Not), |_, rhs, extra| {
            let expk = ClExpressionKind::UnaryOp(ClUnaryOp::new(UnaryOperator::Neg, Box::new(rhs)));
            ClExpression::new(expk, extra.span())
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

pub fn parse_identifier<'a, I>() -> impl Parser<'a, I, Ident, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    select! { Token::Ident(s) => s }
        .map_with(|ident, extra| Ident::new(ident, extra.span()))
        .labelled("identifier")
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
        .map(|((e1, e2), e3)| IfStm::new(Box::new(e1), Box::new(e2), Box::new(e3)))
        .labelled("if stm")
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

    parse_identifier()
        .then(parse_args)
        .map(|(func_ident, params)| FuncCall::new(func_ident, params))
        .labelled("function call")
}

fn parse_compound_expression<'a, I>(
    item: impl Parser<'a, I, ClItem, ParserErr<'a>> + Clone,
    expr: impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone,
) -> impl Parser<'a, I, ClCompoundExpression, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    just(Token::LBrace)
        .ignore_then(item.repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map(|mut items| {
            let final_expr = items
                .pop_if(|item| item.is_expression())
                .map(|e| Box::new(e.get_exp().clone()));
            ClCompoundExpression::new(items, final_expr)
        })
}

pub fn parse_expression<'a, I>(
    item: impl Parser<'a, I, ClItem, ParserErr<'a>> + Clone + 'a,
) -> impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    recursive(|rec| {
        let bracketed_expr = rec
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        choice((
            parse_compound_expression(item.clone(), rec.clone()).map_with(|comp, extra| {
                ClExpression::new(ClExpressionKind::Block(comp), extra.span())
            }),
            parse_function_call(rec.clone()).map_with(|i, extra| {
                ClExpression::new(ClExpressionKind::FunctionCall(i), extra.span())
            }),
            parse_binary_unary_ops(rec.clone()),
            parse_if(rec.clone())
                .map_with(|i, extra| ClExpression::new(ClExpressionKind::IfStm(i), extra.span())),
            parse_literal()
                .map_with(|i, extra| ClExpression::new(ClExpressionKind::Literal(i), extra.span())),
            parse_identifier().map_with(|i, extra| {
                ClExpression::new(ClExpressionKind::Identifier(i), extra.span())
            }),
            bracketed_expr,
        ))
    })
}
