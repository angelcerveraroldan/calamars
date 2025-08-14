//! Parser for Calamars

use chumsky::{input::ValueInput, prelude::*};

use crate::token::Token;

pub trait TokenInput<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
impl<'a, I> TokenInput<'a> for I where I: ValueInput<'a, Token = Token, Span = SimpleSpan> {}

struct Ident {
    src: String,
    span: SimpleSpan,
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
pub fn base_type_parser<'a, I>()
-> impl Parser<'a, I, ClLiteral, extra::Err<Rich<'a, Token>>> + Clone
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

pub enum ClType {
    /// Basic / standard types such as Int, String, Char, Real, ...
    /// as well as types that require many segments, such as people.Person
    Path {
        segments: Vec<Ident>,
        total_span: SimpleSpan,
    },
    /// An array of some type such as [Int]
    Array {
        elem_type: Box<Self>,
        total_span: SimpleSpan,
    },
    /// A function (I1, I2, I3, ...) -> (O1, O2, O3, ...)
    Func {
        inputs: Vec<Self>,
        output: Box<Self>,
        total_span: SimpleSpan,
    },
}

fn parse_cltype_annotation<'a, I>()
-> impl Parser<'a, I, ClLiteral, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    // TODO
    parse_atom()
}
