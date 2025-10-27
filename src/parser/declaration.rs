use std::path::Iter;

use crate::parser::expression::{parse_expression, parse_identifier};
use crate::parser::{ParserErr, TokenInput, parse_cltype_annotation, parse_semicolon};
use crate::syntax::ast::*;
use crate::syntax::token::Token;

use chumsky::{input::ValueInput, prelude::*};

/// Try to parse `: <type>`. If the type was not defined (`var x = 2;`) then None will be returned.
///
/// The None type should be handled by the semantics checks, not the parser.
fn parse_maybe_type<'a, I>() -> impl Parser<'a, I, Type, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    (just(Token::Colon).ignore_then(parse_cltype_annotation()))
        .or_not()
        .map(|x| match x {
            Some(ty) => ty,
            None => Type::Error,
        })
}

pub fn parse_import<'a, I>() -> impl Parser<'a, I, Import, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let parse_idents = parse_identifier()
        .separated_by(just(Token::Dot))
        .at_least(1)
        .collect::<Vec<Ident>>();

    just(Token::Import)
        .ignore_then(parse_idents)
        .then_ignore(parse_semicolon())
        .map_with(|idents, extra| Import::new(idents, extra.span()))
}

fn parse_binding<'a, I>(
    item: impl Parser<'a, I, Item, ParserErr<'a>> + Clone + 'a,
) -> impl Parser<'a, I, Binding, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let var_or_val = (just(Token::Var).map(|_| true)).or(just(Token::Val).map(|_| false));

    var_or_val
        .then(parse_identifier())
        .then(parse_maybe_type())
        .then_ignore(just(Token::Equal))
        .then(parse_expression(item.clone()))
        .then_ignore(parse_semicolon())
        .map_with(|(((mutable, vname), vtype), assigned), extra| {
            Binding::new(vname, vtype, Box::new(assigned), mutable, extra.span())
        })
        .labelled("val/var declaration")
}

/// Parser the inputs for a function declaration
///
/// This parser handles `(x: Int, y: String)` in `def foo(x: Int, y: String) = ...`
///
/// If the user mistakenly didn't give one of the variables a type, this will still parse, but with
/// a None type. This error should be handled in the semantic check.
///
/// Doing this allows for better error recovety.
fn parse_func_input<'a, I>() -> impl Parser<'a, I, Vec<(Ident, Type)>, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let name_type = parse_identifier().then(parse_maybe_type());

    name_type
        .separated_by(just(Token::Comma))
        .collect::<Vec<(Ident, Type)>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
        .labelled("Function input")
}

fn parse_func_declaration<'a, I>(
    item: impl Parser<'a, I, Item, ParserErr<'a>> + Clone + 'a,
) -> impl Parser<'a, I, FuncDec, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let doc_comment = select! { Token::DocComment(s) => s }.or_not();

    let funcp = just(Token::Def)
        .ignore_then(parse_identifier()) // Function name
        .then(parse_func_input()) // Input types, and names
        .then(parse_maybe_type().map(|x| match x {
            Type::Error => Type::Unit,
            _ => x,
        })) // Output type
        .then_ignore(just(Token::Equal))
        .then(parse_expression(item.clone())) // Body of the funcion
        .map_with(|(((fname, inputs), out_type), body), extra| {
            FuncDec::new(fname, inputs, out_type, body, extra.span(), None)
        })
        .labelled("function declaration");

    doc_comment // Perhaps parse the documentation for this function
        .then(funcp)
        .map(|(comment, mut func)| {
            func.doc_comment = comment;
            func
        })
}

pub fn parse_cldeclaration<'a, I>(
    item: impl Parser<'a, I, Item, ParserErr<'a>> + Clone + 'a,
) -> impl Parser<'a, I, Declaration, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    choice((
        parse_func_declaration(item.clone()).map(Declaration::Function),
        parse_binding(item.clone()).map(Declaration::Binding),
    ))
}
