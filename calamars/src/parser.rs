//! Parser for Calamars

use chumsky::{input::ValueInput, prelude::*};
use proptest::array;

use crate::token::Token;

pub trait TokenInput<'a>: ValueInput<'a, Token = Token, Span = SimpleSpan> {}
impl<'a, I> TokenInput<'a> for I where I: ValueInput<'a, Token = Token, Span = SimpleSpan> {}

type Ident = String;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ClExpression {
    Literal(ClLiteral),
    Type(ClType),
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

/// Parse any base value, including nested arrays
pub fn calamars_parser<'a, I>()
-> impl Parser<'a, I, ClExpression, extra::Err<Rich<'a, Token>>> + Clone
where
    I: TokenInput<'a>,
{
    choice((
        parse_cltype_annotation().map(ClExpression::from),
        parse_base_type().map(ClExpression::from),
    ))
}
