use std::io::{Write, stdin, stdout};

use parlib::{
    parsers::{
        ParseMatch, ParseWhile, ParseWhileOrNothing, and_p::KeepFirstOutputOnly,
        string_p::StringParser,
    },
    traits::Parser,
};

#[derive(Debug)]
pub enum Prim {
    Str(String),
    Int(i64),
    Float(f64),
}

fn int_p() -> impl Parser<Output = Prim> {
    ParseWhile(|c| c.is_numeric())
        .with_mapping(&|numb| Prim::Int(numb.parse::<i64>().unwrap()))
        .with_error("Error parsing Primitive Integer, expected to find at least one number.")
}

fn float_p() -> impl Parser<Output = Prim> {
    ParseWhile(|c| c.is_numeric())
        .with_error("Error parsing Floating Point, expected to find at least one number.")
        .and_then(ParseMatch(".").with_error("Error parsing Floating Point, expected to find '.'"))
        .combine(KeepFirstOutputOnly)
        .and_then(ParseWhileOrNothing(|c| c.is_numeric()))
        .with_mapping(&|(a, b)| {
            let f = format!("{a}.{b}").parse::<f64>().unwrap();
            Prim::Float(f)
        })
}

fn prim_p() -> impl Parser<Output = Prim> {
    StringParser
        .with_mapping(&|s| Prim::Str(s))
        .otherwise(float_p())
        .otherwise(int_p())
}

fn main() {
    loop {
        println!("Enter a single line of calamrs");
        let _ = stdout().flush();
        let mut buffer: String = String::new();
        stdin()
            .read_line(&mut buffer)
            .expect("Error reading user input");
        let par = prim_p();
        println!("{:?}", par.parse(&buffer.into()));
    }
}
