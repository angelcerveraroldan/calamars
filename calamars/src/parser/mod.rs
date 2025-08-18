//! Parser for Calamars

pub mod declaration;
pub mod expression;

use chumsky::{input::ValueInput, prelude::*};

use crate::{
    parser::{
        declaration::{ClDeclaration, parse_cldeclaration},
        expression::{ClExpression, parse_expression},
    },
    token::Token,
};

pub trait TokenInput<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
impl<'a, I> TokenInput<'a> for I where I: ValueInput<'a, Token = Token, Span = SimpleSpan> {}

type ParserErr<'a> = extra::Err<Rich<'a, Token>>;
type Ident = String;

/// A Calamars module / file
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Module {
    items: Vec<ClItem>,
}

pub fn parse_module<'a, I>() -> impl Parser<'a, I, Module, ParserErr<'a>> + Clone
where
    I: TokenInput<'a>,
{
    parse_cl_item()
        .repeated()
        .collect::<Vec<_>>()
        .map(|items| Module { items })
}

/// Any one thing in the Cl language
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ClItem {
    Declaration(ClDeclaration),
    Expression(ClExpression),

    // TODO:
    Import,
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
pub fn parse_literal<'a, I>() -> impl Parser<'a, I, ClLiteral, ParserErr<'a>> + Clone
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
        }
        .labelled("literal");

        let arr = value
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .labelled("array");

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

fn parse_cltype_path<'a, I>() -> impl Parser<'a, I, ClType, ParserErr<'a>> + Clone
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
