use crate::parser::expression::parse_expression;
use crate::parser::{
    ClType, Ident, ParserErr, TokenInput, expression::ClExpression, parse_cltype_annotation,
};

use crate::token::Token;

use chumsky::{input::ValueInput, prelude::*};

pub enum ClDeclaration {
    Binding(ClBinding),
    Function(ClFuncDec),
}

/// Value and Variable declaration
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ClBinding {
    vname: Ident,
    vtype: ClType,
    assigned: Box<ClExpression>,
    mutable: bool,
}

fn parse_binding<'a, I>() -> impl Parser<'a, I, ClBinding, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let var_or_val = (just(Token::Var).map(|_| true)).or(just(Token::Val).map(|_| false));

    var_or_val
        .then(select! {Token::Ident(valname) => valname})
        .then_ignore(just(Token::Colon))
        .then(parse_cltype_annotation())
        .then_ignore(just(Token::Equal))
        .then(parse_expression())
        .then_ignore(just(Token::Semicolon))
        .map(|(((mutable, vname), vtype), assigned)| ClBinding {
            vname,
            vtype,
            assigned: Box::new(assigned),
            mutable,
        })
        .labelled("val/var declaration")
}

pub struct ClFuncDec {
    fname: Ident,
    inputs: Vec<(Ident, ClType)>,
    out_type: ClType,
    body: ClExpression,
}

fn parse_func_input<'a, I>() -> impl Parser<'a, I, Vec<(Ident, ClType)>, ParserErr<'a>>
where
    I: TokenInput<'a>,
{
    let name_type = select! { Token::Ident(s) => s }
        .then_ignore(just(Token::Colon))
        .then(parse_cltype_annotation());

    name_type
        .separated_by(just(Token::Comma))
        .collect::<Vec<(Ident, ClType)>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
}

fn parse_func_declaration<'a, I>() -> impl Parser<'a, I, ClFuncDec, ParserErr<'a>>
where
    I: TokenInput<'a>,
{
    just(Token::Def)
        .ignore_then(select! { Token::Ident(s) => s}) // Function name
        .then(parse_func_input()) // Input types, and names
        .then_ignore(just(Token::Colon))
        .then(parse_cltype_annotation()) // Output type
        .then_ignore(just(Token::Equal))
        .then(parse_expression()) // Body of the funcion
        .map(|(((fname, inputs), out_type), body)| ClFuncDec {
            fname,
            inputs,
            out_type,
            body,
        })
        .labelled("function declaration")
}

pub fn parse_cldeclaration<'a, I>() -> impl Parser<'a, I, ClDeclaration, ParserErr<'a>>
where
    I: TokenInput<'a>,
{
    choice((
        parse_func_declaration().map(ClDeclaration::Function),
        parse_binding().map(ClDeclaration::Binding),
    ))
}
