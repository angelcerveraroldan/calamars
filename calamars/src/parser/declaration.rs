use crate::parser::{
    ClType, Ident, ParserErr, TokenInput, expression::ClExpression, parse_cltype_annotation,
};

use crate::token::Token;

use chumsky::{input::ValueInput, prelude::*};

/// Value and Variable declaration
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct ClBinding {
    vname: Ident,
    vtype: ClType,
    assigned: Box<ClExpression>,
    mutable: bool,
}

fn parse_binding<'a, I>(
    expr: impl Parser<'a, I, ClExpression, ParserErr<'a>> + Clone,
) -> impl Parser<'a, I, ClBinding, ParserErr<'a>> + Clone
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
        .labelled("val/var assignment")
}
