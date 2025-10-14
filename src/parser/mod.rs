//! The calamars parser

pub mod declaration;
pub mod expression;

use chumsky::{input::ValueInput, prelude::*};

use crate::{
    parser::{
        declaration::parse_cldeclaration,
        expression::{parse_expression, parse_identifier},
    },
    syntax::{ast::*, token::Token},
};

pub fn parse_module<'a, I>() -> impl Parser<'a, I, Module, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    parse_cl_item()
        .repeated()
        .collect::<Vec<_>>()
        .map(|items| Module { items })
}

impl ClItem {
    pub fn is_expression(&self) -> bool {
        match self {
            ClItem::Expression(cl_expression) => true,
            _ => false,
        }
    }

    pub fn get_exp(&self) -> &ClExpression {
        match self {
            ClItem::Expression(cl_expression) => cl_expression,
            _ => panic!("Cannot get expression of non-expression type"),
        }
    }
}

pub fn parse_cl_item<'a, I>() -> impl Parser<'a, I, ClItem, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    recursive(|item| {
        choice((
            parse_cldeclaration(item.clone()).map(ClItem::Declaration),
            parse_expression(item.clone()).map(ClItem::Expression),
        ))
    })
}

/// Parse any base value, including nested arrays
pub fn parse_literal<'a, I>() -> impl Parser<'a, I, ClLiteral, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    recursive(|value| {
        let atom = select! {
            Token::True      => ClLiteralKind::Boolean(true),
            Token::False     => ClLiteralKind::Boolean(false),
            Token::Int(i)    => ClLiteralKind::Integer(i),
            Token::Float(f)  => ClLiteralKind::Real(f),
            Token::String(s) => ClLiteralKind::String(s),
            Token::Char(c)   => ClLiteralKind::Char(c),
        }
        .map_with(|kind, extra| ClLiteral::new(kind, extra.span()))
        .labelled("literal");

        let arr = value
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map_with(|arr, extra| {
                let cl_lkind = ClLiteralKind::Array(arr);
                ClLiteral::new(cl_lkind, extra.span())
            })
            .labelled("array");

        choice((atom, arr))
    })
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

fn parse_cltype_path<'a, I>() -> impl Parser<'a, I, ClType, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    // Parse just an ident
    let ident_p = parse_identifier();

    ident_p
        .clone()
        .then(just(Token::Dot).ignore_then(ident_p).repeated().collect())
        .map(|(head, tail): (Ident, Vec<Ident>)| {
            let mut tmp = Vec::with_capacity(tail.len() + 1);
            tmp.push(head);
            tmp.extend(tail);
            ClType::Path { segments: tmp }
        })
}

// TODO: Maybe its better to have (x, y, z) be a Tuple type, then functions are always A->B, where
// A and B are standard types ...
fn parse_cltype_annotation<'a, I>() -> impl Parser<'a, I, ClType, ParserErr<'a>> + Clone
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
    .labelled("type annotation")
}
