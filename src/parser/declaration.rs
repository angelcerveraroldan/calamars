use crate::parser::expression::{parse_expression, parse_identifier};
use crate::parser::parse_cltype_annotation;

use crate::syntax::ast::*;
use crate::syntax::token::Token;

use chumsky::{input::ValueInput, prelude::*};

/// Try to parse `: <type>`. If the type was not defined (`var x = 2;`) then None will be returned.
///
/// The None type should be handled by the semantics checks, not the parser.
fn parse_maybe_type<'a, I>() -> impl Parser<'a, I, Option<ClType>, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    (just(Token::Colon).ignore_then(parse_cltype_annotation())).or_not()
}

fn parse_binding<'a, I>(
    item: impl Parser<'a, I, ClItem, ParserErr<'a>> + Clone + 'a,
) -> impl Parser<'a, I, ClBinding, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let var_or_val = (just(Token::Var).map(|_| true)).or(just(Token::Val).map(|_| false));

    var_or_val
        .then(parse_identifier())
        .then(parse_maybe_type())
        .then_ignore(just(Token::Equal))
        .then(parse_expression(item.clone()))
        .then_ignore(just(Token::Semicolon))
        .map_with(|(((mutable, vname), vtype), assigned), extra| {
            ClBinding::new(vname, vtype, Box::new(assigned), mutable, extra.span())
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
fn parse_func_input<'a, I>()
-> impl Parser<'a, I, Vec<(Ident, Option<ClType>)>, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let name_type = parse_identifier().then(parse_maybe_type());

    name_type
        .separated_by(just(Token::Comma))
        .collect::<Vec<(Ident, Option<ClType>)>>()
        .delimited_by(just(Token::LParen), just(Token::RParen))
}

fn parse_func_declaration<'a, I>(
    item: impl Parser<'a, I, ClItem, ParserErr<'a>> + Clone + 'a,
) -> impl Parser<'a, I, ClFuncDec, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    let doc_comment = select! { Token::DocComment(s) => s }.or_not();

    doc_comment // Perhaps parse the documentation for this function
        .then_ignore(just(Token::Def).labelled("'Function declaration after doc comment'"))
        .then(parse_identifier()) // Function name
        .then(parse_func_input()) // Input types, and names
        .then(parse_maybe_type()) // Output type
        .then_ignore(just(Token::Equal))
        .then(parse_expression(item.clone())) // Body of the funcion
        .map_with(|((((comment, fname), inputs), out_type), body), extra| {
            ClFuncDec::new(fname, inputs, out_type, body, extra.span(), comment)
        })
        .labelled("function declaration")
}

pub fn parse_cldeclaration<'a, I>(
    item: impl Parser<'a, I, ClItem, ParserErr<'a>> + Clone + 'a,
) -> impl Parser<'a, I, ClDeclaration, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    choice((
        parse_func_declaration(item.clone()).map(ClDeclaration::Function),
        parse_binding(item.clone()).map(ClDeclaration::Binding),
    ))
}
